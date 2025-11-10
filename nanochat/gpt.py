"""
coding = utf-8
Licensed under "MIT License"
Commercial use is of course permitted
SEA Model Op.0: Saint Iberis PyTorch implementation
(Saint Iberis is the early experimental model)

Saint Iberis = SLC2 + GPT

GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Multi-Query Attention (MQA) support for more efficient inference
"""

import math
from functools import partial
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Any, Optional
from einops import rearrange

from nanochat.common import get_dist_info, print0
from nanochat.muon import Muon, DistMuon
from nanochat.adamw import DistAdamW


@dataclass
class GPTConfig:
    sequence_len: int = 1024
    vocab_size: int = 50304
    n_layer: int = 16
    n_head: int = 6 # number of query heads
    n_kv_head: int = 6 # number of key/value heads (MQA)
    n_embd: int = 768
    n_kernel: int = 3
    slc_rate: int = 3

def norm(x):
    # Purely functional rmsnorm with no learnable params
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last time into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    out = torch.cat([y1, y2], 3) # re-assemble
    out = out.to(x.dtype) # ensure input/output dtypes match
    return out

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x, cos_sin, kv_cache):
        B, T, C = x.size()

        # Project the input to get queries, keys, and values
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin) # QK rotary embedding
        q, k = norm(q), norm(k) # QK norm
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2) # make head be batch dim, i.e. (B, T, H, D) -> (B, H, T, D)

        # Apply KV cache: insert current k,v into cache, get the full view so far
        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)
        Tq = q.size(2) # number of queries in this forward pass
        Tk = k.size(2) # number of keys/values in total (in the cache + current forward pass)

        # Attention: queries attend to keys/values autoregressively. A few cases to handle:
        enable_gqa = self.n_head != self.n_kv_head # Group Query Attention (GQA): duplicate key/value heads to match query heads if desired
        if kv_cache is None or Tq == Tk:
            # During training (no KV cache), attend as usual with causal attention
            # And even if there is KV cache, we can still use this simple version when Tq == Tk
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)
        elif Tq == 1:
            # During inference but with a single query in this forward pass:
            # The query has to attend to all the keys/values in the cache
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)
        else:
            # During inference AND we have a chunk of queries in this forward pass:
            # First, each query attends to all the cached keys/values (i.e. full prefix)
            attn_mask = torch.zeros((Tq, Tk), dtype=torch.bool, device=q.device) # True = keep, False = mask
            prefix_len = Tk - Tq
            if prefix_len > 0: # can't be negative but could be zero
                attn_mask[:, :prefix_len] = True
            # Then, causal attention within this chunk
            attn_mask[:, prefix_len:] = torch.tril(torch.ones((Tq, Tq), dtype=torch.bool, device=q.device))
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, enable_gqa=enable_gqa)

        # Re-assemble the heads side by side and project back to residual stream
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


# Inference Cache class for SLC2
class SLCInferenceCache:
    __slots__ = ("conv_state", "index", "kernel_size")

    def __init__(self, conv_state: torch.Tensor, index: int = 0):
        # conv_state: (batch, hidden_size, kernel_size)
        self.conv_state = conv_state
        self.index = int(index)
        self.kernel_size = conv_state.size(-1)

    @staticmethod
    def alloc(
        batch_size: int,
        hidden_size: int,
        kernel_size: int = 5,
        device: Any = None,
        dtype: Any = None
    ):
        return SLCInferenceCache(
            conv_state=torch.zeros(
                batch_size,
                hidden_size,
                kernel_size,
                device=device,
                dtype=dtype
            ), index=0
        )

    def clear_(self):
        self.conv_state.zero_()
        self.index = 0


# SLC2 implementation
# Copyright 2025 Rikka Botan. All rights reserved
class SLC2(nn.Module):
    def __init__(
        self,
        config
    ):
        """
        ## Substitution Liquid Convolution Module

        inspired by LFM2.LFM2ConvBlock
        ```
        Formulation:

        x ∈ ℝ^{B×S×E}
        y ∈ ℝ^{B×S×E}

        y = B ⋅ ∏ᵢ₌ⱼ⁽ʲ⁺ᵏ⁾ Aᵢ ⋅ xᵢ

        ----------------------------------------
        Algorithm: SLC2
        ----------------------------------------
        Input: x: (B, S, E)
        Output: y: (B, S, E)
            1: A, B, x₁: (B, S, E) <- Linear(x)
            2: x₂: (B, S, E) <- Convolution1D(E, E)(SiLU(A)*x₁)
            3: x₃: (B, S, E) <- B*SiLU(x₂)
            4: y: (B, S, E) <- Linear(x₃)
            5: return y
        ----------------------------------------
        ```
        """
        super().__init__()
        self.n_embed = config.n_embd
        self.n_head = config.n_head
        self.d_head = config.n_embd//config.n_head
        self.n_kernel = config.n_kernel
        self.x_proj = nn.Linear(
            in_features=self.n_embed,
            out_features=self.n_embed,
            bias=False
        )
        self.alpha_proj = nn.Linear(
            in_features=self.n_embed,
            out_features=self.n_head,
            bias=False
        )
        self.A_proj = nn.Linear(
            in_features=self.n_embed,
            out_features=self.d_head,
            bias=False
        )
        self.B_proj = nn.Linear(
            in_features=self.n_embed,
            out_features=self.n_embed,
            bias=False
        )
        self.conv1d = nn.Conv1d(
            in_channels=self.n_embed,
            out_channels=self.n_embed,
            kernel_size=self.n_kernel,
            stride=1,
            padding=self.n_kernel-1,
            dilation=1,
            groups=self.n_embed,
            bias=False,
            padding_mode="zeros"
        )
        self.c_proj = nn.Linear(
            in_features=self.n_embed,
            out_features=self.n_embed,
            bias=False
        )

        self.cache: Optional[SLCInferenceCache] = None
    
    def alloc_cache(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        self.cache = SLCInferenceCache.alloc(
            batch_size,
            self.n_embed,
            self.n_kernel,
            device=device,
            dtype=dtype
        )
        self.cache.conv_state = self.cache.conv_state.contiguous()

    def clear_cache(
        self
    ):
        if self.cache is not None:
            self.cache.clear_()

    def forward(
        self,
        hidden_states: torch.Tensor,
        use_cache: bool = False
    ) -> torch.Tensor:
        bsz, seql, _ = hidden_states.size()
        if seql > 1 or not use_cache:
            x = self.x_proj(hidden_states)
            alpha = self.alpha_proj(hidden_states)
            A = self.A_proj(hidden_states)
            B = self.B_proj(hidden_states)
            A = A.unsqueeze(-2) * F.silu(alpha).unsqueeze(-1)
            xA = self.conv1d(
            (F.silu(A.reshape(bsz, seql, -1)) * x ).transpose(1, 2)).transpose(1, 2)
            xA = F.silu(xA[:, :seql])
            xAB = B * xA
            y = self.c_proj(xAB)
            return y

        if self.cache is None or self.cache.conv_state.size(0) != bsz:
            self.alloc_cache(
                bsz,
                device=hidden_states.device,
                dtype=hidden_states.dtype
            )

        hidden_states, prefix = hidden_states[:, -1], hidden_states[:, :-1]
        y_t = self.step(hidden_states, self.cache)
        y = torch.cat([prefix, y_t.unsqueeze(1)], dim=1)
        return y

    def step(
        self,
        hidden_states: torch.Tensor,
        cache: SLCInferenceCache
    ) -> torch.Tensor:
        bsz = hidden_states.size(0)
        x = self.x_proj(hidden_states)
        alpha = self.alpha_proj(hidden_states)
        A = self.A_proj(hidden_states)
        B = self.B_proj(hidden_states)
        A = A.unsqueeze(-2) * F.silu(alpha).unsqueeze(-1)
        xA = F.silu(A.reshape(bsz, 1, -1)) * x
        cache.conv_state.copy_(
            torch.roll(cache.conv_state, shifts=-1, dims=-1))
        cache.conv_state[:, :, -1] = xA.squeeze(1)
        xA = torch.sum(
            cache.conv_state 
            * rearrange(self.conv1d.weight, "d 1 w -> d w"), 
            dim=-1
        ) # (B D)
        if self.conv1d.bias is not None:
            xA = xA + self.conv1d.bias
        xAB = B * F.silu(xA)
        y_t = self.c_proj(xAB)

        return y_t


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        use_cache: bool = False
    ) -> torch.Tensor:
        if use_cache:
            x, prefix = x[:, -1], x[:, :-1]
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        if use_cache:
            x = torch.cat([prefix, x.unsqueeze(1)], dim=1)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        if layer_idx%config.slc_rate==1 or layer_idx==config.n_layer-1:
            self.attn = CausalSelfAttention(config, layer_idx)
            self.attn_mode = True
        else:
            self.attn = SLC2(config)
            self.attn_mode = False
            
        self.mlp = MLP(config)

    def forward(self, x, cos_sin, kv_cache, use_cache=False):
        if self.attn_mode:
            x = x + self.attn(norm(x), cos_sin, kv_cache)
        else:
            x = x + self.attn(norm(x), use_cache)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # To support meta device initialization, we init the rotary embeddings here, but it's fake
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        self.rotary_seq_len = config.sequence_len * 10 # 10X over-compute should be enough, TODO make nicer?
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)

    def init_weights(self):
        self.apply(self._init_weights)
        # zero out classifier weights
        torch.nn.init.zeros_(self.lm_head.weight)
        # zero out c_proj weights in all blocks
        for block in self.transformer.h:
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
        # init the rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        # Cast the embeddings from fp32 to bf16: optim can tolerate it and it saves memory: both in the model and the activations
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # https://arxiv.org/pdf/2310.17813
            fan_out = module.weight.size(0)
            fan_in = module.weight.size(1)
            std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=1.0)

    # TODO: bump base theta more, e.g. 100K is more common more recently
    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        # autodetect the device from model embeddings
        if device is None:
            device = self.transformer.wte.weight.device
        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16() # keep them in bfloat16
        cos, sin = cos[None, :, None, :], sin[None, :, None, :] # add batch and head dims for later broadcasting
        return cos, sin

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        """ Return the estimated FLOPs per token for the model. Ref: https://arxiv.org/abs/2204.02311 """
        nparams = sum(p.numel() for p in self.parameters())
        nparams_embedding = self.transformer.wte.weight.numel()
        l, h, q, t = self.config.n_layer, self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        num_flops_per_token = 6 * (nparams - nparams_embedding) + 12 * l * h * q * t
        return num_flops_per_token

    def setup_optimizers(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()
        # Separate out all parameters into 3 groups (matrix, embedding, lm_head)
        embedding_params = list(self.transformer.wte.parameters())
        matrix_params = []
        matrix_1d_params = []
        for p in self.transformer.h.parameters():
            if not p.requires_grad:
                continue
            # Muon は 2D (ndim == 2) のみ期待するのでフィルタ
            if p.ndim == 2:
                matrix_params.append(p)
            else:
                matrix_1d_params.append(p)
        # matrix_params = list(self.transformer.h.parameters())
        lm_head_params = list(self.lm_head.parameters())
        assert len(list(self.parameters())) == len(matrix_params) + len(matrix_1d_params) + len(embedding_params) + len(lm_head_params)
        # Create the AdamW optimizer for the embedding and lm_head
        # Scale the LR for the AdamW parameters by ∝1/√dmodel (having tuned the LRs for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        if rank == 0:
            print(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")
        adam_groups = [
            dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale),
            dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),
            dict(params=matrix_1d_params, lr=matrix_lr * dmodel_lr_scale),
        ]
        adamw_kwargs = dict(betas=(0.8, 0.95), eps=1e-10, weight_decay=weight_decay)
        AdamWFactory = DistAdamW if ddp else partial(torch.optim.AdamW, fused=True)
        adamw_optimizer = AdamWFactory(adam_groups, **adamw_kwargs)
        # Create the Muon optimizer for the linear layers
        muon_kwargs = dict(lr=matrix_lr, momentum=0.95)
        MuonFactory = DistMuon if ddp else Muon
        muon_optimizer = MuonFactory(matrix_params, **muon_kwargs)
        # Combine them the two optimizers into one list
        optimizers = [adamw_optimizer, muon_optimizer]
        for opt in optimizers:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]
        return optimizers

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean', use_cache=False):
        B, T = idx.size()

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim))
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == torch.bfloat16, "Rotary embeddings must be in bfloat16"
        # if kv cache exists, we need to offset the rotary embeddings to the current position in the cache
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T] # truncate cache to current sequence length

        # Forward the trunk of the Transformer
        x = self.transformer.wte(idx)
        x = norm(x)
        for block in self.transformer.h:
            x = block(x, cos_sin, kv_cache, use_cache)
        x = norm(x)

        # Forward the lm_head (compute logits)
        softcap = 15
        if targets is not None:
            # training mode: compute and return the loss
            # TODO: experiment with Liger Kernels / chunked cross-entropy etc.
            logits = self.lm_head(x)
            logits = softcap * torch.tanh(logits / softcap) # logits softcap
            logits = logits.float() # use tf32/fp32 for logits
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            return loss
        else:
            # inference mode: compute and return the logits
            logits = self.lm_head(x)
            logits = softcap * torch.tanh(logits / softcap) # logits softcap
            return logits

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.LongTensor = None,
        use_cache: bool = True,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        top_k: int = 20,
        top_p: float = 0.95,
        repetition_penalty: float = 1.15,
        eos_token_id: int = 2
    ):
        generated = input_ids.clone().to(input_ids.device)
        bsz = input_ids.size(0)
        tokens = input_ids
        prefix, tokens = input_ids[:, :-1], input_ids[:, -1:]
        _ = self(prefix)

        # Generate
        for _ in range(max_new_tokens):
            with torch.no_grad():
                out = self(tokens, use_cache=use_cache)
            logits = out[:, -1]
            if repetition_penalty != 1.0:
                for b in range(bsz):
                    u = torch.unique(generated[b])
                    token_logits = logits[b, u]
                    lt_mask = token_logits < 0
                    gt_mask = ~lt_mask
                    if gt_mask.any():
                        logits[b, u[gt_mask]] /= repetition_penalty
                    if lt_mask.any():
                        logits[b, u[lt_mask]] *= repetition_penalty
            if temperature != 1.0:
                logits = logits / temperature
            if top_k > 0:
                top_k = min(top_k, logits.size(-1))
                threshold = torch.topk(logits, top_k, dim=-1).values[:, -1].unsqueeze(-1)
                logits = logits.masked_fill(logits < threshold, -float("inf"))
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cum_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = False
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[indices_to_remove] = -torch.inf
            probs = F.softmax(logits, dim=-1)
            next_ids = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat((tokens, next_ids), dim=1)
            token = next_ids.item()
            if token == eos_token_id:
                break
            yield token
