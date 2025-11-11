# 🌸 Liquid-Time-nanochat

![Nanochat_saint_iberis](https://github.com/user-attachments/assets/2af52f08-6b04-41ff-8e74-1144e2ec7e9a)

This repository employs a module called **SLC2**, inspired by **Liquid Time-Constant Networks (LTCs)** and **Liquid Foundation Models (LFM2)**, to enable faster training and inference for **nanochat**.
The **SEA Model series Op.0: Saint Iberis** achieves comparable performance while reducing training time by more than **30 minutes** and lowering computational costs by over **$10**.
You are free to use the model from this repository

このリポジトリはnanochatをより高速に学習・推論するために、LTCsおよびLFM2から着想を得たSLC2というモジュールを使用しています。
SEA Model series Op.0: Saint Iberisは元のnanoGPTと比較して学習時間を30分以上、$10以上のコストを削減しながら、同等の性能を達成することが可能です。
モデルはこのリポジトリからご自由に利用できます。

# 🌸 Saint Iberis Architecture

<img width="4400" height="1595" alt="Saint_Iberis" src="https://github.com/user-attachments/assets/08315549-988f-48ad-b58b-a068e1f851dd" />


| Property              | Saint Iberis d20              | Remarks                                               |
| --------------------- | ----------------------------- |------------------------------------------------------ |
| **Total parameters**  | 542,035,200 (542M)            | n_layer: 20, n_head: 10, n_kv_head: 10, n_embd: 1280  |
| **Layers**            | 20 (13 slc2 + 7 attn)         | attn layers: 1, 4, 7, 10, 13, 16, 19                  |
| **Vocabulary size**   | 65,536                        | -                                                     |
| **Training budget**   | 100B tokens                   | Fineweb edu                                           |
| **License**           | MIT                           | -                                                    |

# 🌸 SLC2 Formulation

```math
y = B \odot \Pi_{i=j}^{j+k} A_i \cdot x_i
```

# 🌸 SLC2 pseudo code

```python
----------------------------------------
Algorithm: SLC2
----------------------------------------
Input: x: (B, S, E)
Output: y: (B, S, E)
    1: alpha, A, B, x₁ <- Linear(x)
    2: x₂: (B, S, E) <- Convolution1D(E, E)(SiLU(alpha)*A*x₁)
    3: x₃: (B, S, E) <- B*SiLU(x₂)
    4: y: (B, S, E) <- Linear(x₃)
    5: return y
----------------------------------------
```

# 🌸 Quick start

The fastest way to feel the magic is to run the speedrun script [speedrun.sh](speedrun.sh), which trains and inferences the $100 tier of nanochat. On an 8XH100 node at $24/hr, this gives a total run time of about 3.5 hours. Boot up a new 8XH100 GPU box from your favorite provider (e.g. I use and like [Lambda](https://lambda.ai/service/gpu-cloud)), and kick off the training script:

Fisrt, install this repository
```bash
git clone https://github.com/Rikka-Botan/Liquid_Time_nanochat.git
```

Then, run trainings.
```bash
cd
cd Liquid_Time_nanochat
pwd
bash speedrun.sh
```

Alternatively, since the script runs for 4 hours, I like to launch it like this inside a new screen session `speedrun` (and also log output to `speedrun.log`):

```bash
screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh
```

See the [screen cheatsheet](https://gist.github.com/jctosta/af918e1618682638aa82) if you are less familiar. You can watch it go inside the screen session, or detach with `Ctrl-a d` and `tail speedrun.log` to view progress. Now wait 4 hours. Once it's done, you can talk to your LLM via the ChatGPT-like web UI. Make sure again that your local uv virtual environment is active (run `source .venv/bin/activate`), and serve it:

```bash
python -m scripts.chat_web
```

And then visit the URL shown. Make sure to access it correctly, e.g. on Lambda use the public IP of the node you're on, followed by the port, so for example [http://209.20.xxx.xxx:8000/](http://209.20.xxx.xxx:8000/), etc. Then talk to your LLM as you'd normally talk to ChatGPT! Get it to write stories or poems. Ask it to tell you who you are to see a hallucination. Ask it why the sky is blue. Or why it's green. The speedrun is a 4e19 FLOPs capability model so it's a bit like talking to a kindergartener :).

---

<img width="2672" height="1520" alt="image" src="https://github.com/user-attachments/assets/ed39ddf8-2370-437a-bedc-0f39781e76b5" />

---

# 🌸 Weight is now available

[RikkaBotan/nanochat_d20_saint_iberis](https://huggingface.co/RikkaBotan/nanochat_d20_saint_iberis)

# 🌸 Performance

| Metric          |   BASE     |   MID      |   SFT      |   RL       |
|-----------------|------------|------------|------------|------------|
| CORE            |   0.1796   | -          | -          | -          |
| ARC-Challenge   | -          |   0.2910   |   0.2782   | -          |
| ARC-Easy        | -          |   0.3792   |   0.3864   | -          |
| GSM8K           | -          |   0.0341   |   0.0455   | -          |
| HumanEval       | -          |   0.0732   |   0.0549   | -          |
| MMLU            | -          |   0.3146   |   0.3166   | -          |
| ChatCORE        | -          |   0.2348   |   0.2322   | -          |
**Total wall clock time: 3h15m**

# 🌸 Comparison with nanoGPT

| Metric                |   GPT([karpathy/nanochat](https://github.com/karpathy/nanochat))|   Saint Iberis                                  |
|-----------------------|------------------------------------------------------------     |-----------------------------------------------  |
| Total wall clock time |   3h51m                                                         |   **3h15m**                                     |
| ARC-Challenge         |   **0.2807**                                                    |   0.2782                                        |
| ARC-Easy              |   **0.3876**                                                    |   0.3864                                        |
| HumanEval             |   **0.0854**                                                    |   0.0549                                        |
| MMLU                  |   0.3151                                                        |   **0.3166**                                    |
| ChatCORE              |   0.0844                                                        |   **0.2322**                                    |
| Task Average          |   0.1998                                                        |   **0.2190**                                    |

# 🌸 Training result

## Base Training
- Minimum validation bpb: 0.8287
- Final validation bpb: 0.8287

## Mid Training
- Minimum validation bpb: 0.4116

## SFT Training
- Training loss: 0.5825
- Validation loss: 1.0657

# 🌸 Bigger models

Unsurprisingly, $100 is not enough to train a highly performant ChatGPT clone. In fact, LLMs are famous for their multi-million dollar capex. For our purposes, I think there are two more scales of interest. First is the ~$300 tier d26 model (i.e. depth=26) that trains in ~12 hours, which slightly outperforms GPT-2 CORE score. Second is the $1000 tier (~41.6 hours), just because it's a nice round number. But both of these are not yet fully supported and therefore not attached here in the master branch yet.

That said, to give a sense, the example changes needed for the [speedrun.sh](speedrun.sh) file to train a GPT-2 grade model d26 only involve three changes:

```bash
...
# you'll need to download more data shards for pretraining
# get the number of parameters, multiply 20 to get tokens, multiply by 4.8 to get chars,
# divide by 250 million to get number of shards. todo need to improve this...
python -m nanochat.dataset -n 450 &
...
# use --depth to increase model size. to not oom, halve device batch size 32 -> 16:
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=26 --device_batch_size=16
...
# make sure to use the same later during midtraining:
torchrun --standalone --nproc_per_node=8 -m scripts.mid_train -- --device_batch_size=16
```

That's it! The biggest thing to pay attention to is making sure you have enough data shards to train on (the code will loop and do more epochs over the same training set otherwise, decreasing learning speed a bit), and managing your memory/VRAM, primarily by decreasing the `device_batch_size` until things fit (the scripts automatically compensate by increasing the number of gradient accumulation loops, simply turning parallel compute to sequential compute).

And a bit more about computing environments that will run nanochat:

- The code will run just fine on the Ampere 8XA100 GPU node as well, but a bit slower.
- All code will run just fine on even a single GPU by omitting `torchrun`, and will produce ~identical results (code will automatically switch to gradient accumulation), but you'll have to wait 8 times longer.
- If your GPU(s) have less than 80GB, you'll have to tune some of the hyperparameters or you will OOM / run out of VRAM. Look for `--device_batch_size` in the scripts and reduce it until things fit. E.g. from 32 (default) to 16, 8, 4, 2, or even 1. Less than that you'll have to know a bit more what you're doing and get more creative.
- Most of the code is fairly vanilla PyTorch so it should run on anything that supports that - xpu, mps, or etc, but I haven't implemented this out of the box so it might take a bit of tinkering.

# 🌸 Running on CPU / MPS

nanochat can be run on CPU or on MPS (if you're on Macbook), and will automatically try to detect what device is best to run on. You're not going to get too far without GPUs, but at least you'll be able to run the code paths and maybe train a tiny LLM with some patience. For an example of how to make all the run commands much smaller (feel free to tune!), you can refer to [dev/runcpu.sh](dev/runcpu.sh) file. You'll see that I'm essentially restricting all scripts to train smaller models, to run for shorter number of iterations, etc. This functionality is new, slightly gnarly (touched a lot of code), and was merged in this [CPU|MPS PR](https://github.com/karpathy/nanochat/pull/88) on Oct 21, 2025.

# 🌸 Customization

To customize your nanochat, see [Guide: infusing identity to your nanochat](https://github.com/karpathy/nanochat/discussions/139) in Discussions, which describes how you can tune your nanochat's personality through synthetic data generation and mixing that data into midtraining and SFT stages.

Additionally, to add new abilities to nanochat, see [Guide: counting r in strawberry (and how to add abilities generally)](https://github.com/karpathy/nanochat/discussions/164).

# 🌸 Tests

I haven't invested too much here but some tests exist, especially for the tokenizer. Run e.g. as:

```bash
python -m pytest tests/test_rustbpe.py -v -s
```

# 🌸 File structure

```
.
├── LICENSE
├── README.md
├── dev
│   ├── gen_synthetic_data.py       # Example synthetic data for identity
│   ├── generate_logo.html
│   ├── nanochat.png
│   ├── repackage_data_reference.py # Pretraining data shard generation
│   └── runcpu.sh                   # Small example of how to run on CPU/MPS
├── nanochat
│   ├── __init__.py                 # empty
│   ├── adamw.py                    # Distributed AdamW optimizer
│   ├── checkpoint_manager.py       # Save/Load model checkpoints
│   ├── common.py                   # Misc small utilities, quality of life
│   ├── configurator.py             # A superior alternative to argparse
│   ├── core_eval.py                # Evaluates base model CORE score (DCLM paper)
│   ├── dataloader.py               # Tokenizing Distributed Data Loader
│   ├── dataset.py                  # Download/read utils for pretraining data
│   ├── engine.py                   # Efficient model inference with KV Cache
│   ├── execution.py                # Allows the LLM to execute Python code as tool
│   ├── gpt.py                      # The GPT nn.Module Transformer
│   ├── logo.svg
│   ├── loss_eval.py                # Evaluate bits per byte (instead of loss)
│   ├── muon.py                     # Distributed Muon optimizer
│   ├── report.py                   # Utilities for writing the nanochat Report
│   ├── tokenizer.py                # BPE Tokenizer wrapper in style of GPT-4
│   └── ui.html                     # HTML/CSS/JS for nanochat frontend
├── pyproject.toml
├── run1000.sh                      # Train the ~$800 nanochat d32
├── rustbpe                         # Custom Rust BPE tokenizer trainer
│   ├── Cargo.lock
│   ├── Cargo.toml
│   ├── README.md                   # see for why this even exists
│   └── src
│       └── lib.rs
├── scripts
│   ├── base_eval.py                # Base model: calculate CORE score
│   ├── base_loss.py                # Base model: calculate bits per byte, sample
│   ├── base_train.py               # Base model: train
│   ├── chat_cli.py                 # Chat model (SFT/Mid): talk to over CLI
│   ├── chat_eval.py                # Chat model (SFT/Mid): eval tasks
│   ├── chat_rl.py                  # Chat model (SFT/Mid): reinforcement learning
│   ├── chat_sft.py                 # Chat model: train SFT
│   ├── chat_web.py                 # Chat model (SFT/Mid): talk to over WebUI
│   ├── mid_train.py                # Chat model: midtraining
│   ├── tok_eval.py                 # Tokenizer: evaluate compression rate
│   └── tok_train.py                # Tokenizer: train it
├── speedrun.sh                     # Train the ~$100 nanochat d20
├── tasks
│   ├── arc.py                      # Multiple choice science questions
│   ├── common.py                   # TaskMixture | TaskSequence
│   ├── customjson.py               # Make Task from arbitrary jsonl convos
│   ├── gsm8k.py                    # 8K Grade School Math questions
│   ├── humaneval.py                # Misnomer; Simple Python coding task
│   ├── mmlu.py                     # Multiple choice questions, broad topics
│   ├── smoltalk.py                 # Conglomerate dataset of SmolTalk from HF
│   └── spellingbee.py              # Task teaching model to spell/count letters
├── tests
│   └── test_rustbpe.py
└── uv.lock
```

# 🌸 Contributing

nanochat is nowhere near finished. The goal is to improve the state of the art in micro models that are accessible to work with end to end on budgets of < $1000 dollars. Accessibility is about overall cost but also about cognitive complexity - nanochat is not an exhaustively configurable LLM "framework"; there will be no giant configuration objects, model factories, or if-then-else monsters in the code base. It is a single, cohesive, minimal, readable, hackable, maximally-forkable "strong baseline" codebase designed to run start to end and produce a concrete ChatGPT clone and its report card.

Current LLM policy: disclosure. When submitting a PR, please declare any parts that had substantial LLM contribution and that you have not written or that you do not fully understand.

# 🌸 Acknowledgments

I thank [Andrej Karpathy's](https://huggingface.co/karpathy) fullstack llm project to build an LLM, [nanochat](https://github.com/karpathy/nanochat).

I thank the developers of python and pytorch.

I thank all the researchers for their efforts to date.

I thank Japan's high standard of education.

And most of all, thank you for your interest in this repository.

# 🌸 About us

Japanese independent researcher having shy and pampered personality. Twin-tail hair is a charm point. Interested in nlp. Usually using python and C.

<img width="4405" height="2480" alt="RikkaBotan_Logo" src="https://github.com/user-attachments/assets/3e0819a9-b7ab-4966-8089-cd5b67a15871" />

# 🌸 Cite

```bibtex
@misc{nanochat,
  author = {Andrej Karpathy},
  title = {nanochat: The best ChatGPT that $100 can buy},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/karpathy/nanochat}
}
```

# 🌸 License

MIT
