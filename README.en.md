# Transformer from Scratch: From Attention to GPT

[Türkçe](README.md) | [English](README.en.md)

A repository that walks from the attention mechanism through the Transformer block, positional encoding, training a small GPT, and pre-training objectives, using **working code plus step-by-step hand calculations**. Every folder runs on its own, and its `README.md` (written in Turkish) contains the same numbers the script prints to the terminal.

## Contents

| Folder | Topic | Stack |
|---|---|---|
| [`01_seq2seq_dikkat`](01_seq2seq_dikkat) | Seq2seq, the fixed-size context vector bottleneck, additive (Bahdanau) and multiplicative (Luong) attention, alignment matrices | NumPy |
| [`02_olceklenmis_nokta_carpimi`](02_olceklenmis_nokta_carpimi) | Scaled dot-product attention (Q, K, V), why divide by √d_k (softmax saturation), causal mask | NumPy |
| [`03_cok_basli_dikkat`](03_cok_basli_dikkat) | Multi-head self- and cross-attention, `O(n²)` compute/memory cost, whether attention weights explain model decisions | NumPy |
| [`04_transformer_blogu`](04_transformer_blogu) | Residual connections, Post-LN vs Pre-LN, LayerNorm vs RMSNorm, ReLU / GELU / SwiGLU, encoder vs decoder block, parameter counting | NumPy + PyTorch |
| [`05_konum_kodlama`](05_konum_kodlama) | Sinusoidal embeddings, RoPE, ALiBi, length generalization | NumPy |
| [`06_mini_gpt`](06_mini_gpt) | A character-level GPT in PyTorch, training loss, greedy / temperature / top-k / top-p sampling | PyTorch |
| [`07_on_egitim_hedefleri`](07_on_egitim_hedefleri) | GPT, BERT and T5 pre-training objectives, encoder-only / decoder-only / encoder-decoder, the pre-train + fine-tune paradigm | NumPy |

Reading order 01 → 07 is recommended; each folder builds on the previous one.

## Setup

```bash
git clone <this-repo-url>
cd transformer-from-scratch
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

cd 02_olceklenmis_nokta_carpimi
python olceklenmis_dikkat.py
```

01, 02, 03, 05 and 07 need only `numpy` and `matplotlib`. 04 and 06 also need `torch` (CPU is enough; training in 06 takes about 30 seconds on 2 cores). Figures are written to each folder's `figures/`.

## What this repo shows, and what it does not

- **Analytical / verified:** hand calculations match the NumPy output (02, 04, 05); parameter counts match PyTorch layers exactly (04); RoPE's relative-distance property and the sinusoidal linear relation are checked numerically (05).
- **Small experiments:** the bottleneck experiment (01), √d_k scaling (02), the Post-LN vs Pre-LN gradient comparison (04, **single seed, 12 layers**), and sampling strategies (06, **single training run**). They show a direction; they are not reproductions of the papers' results.
- **Hand-built illustrations:** the alignment matrices in 01 and the "heads" figure in 03 are not learned; they are constructed for explanation, and the READMEs say so.
- **06 memorizes.** The training text is only about 2.7k characters, and validation loss rises after roughly step 200. It demonstrates the architecture and sampling behavior, not model quality. Use `--veri` to train on your own text.
- **Not measured:** long-sequence accuracy (05), GPU time/memory (03), the effect of pre-training objectives on real model quality (07). See the original papers for those.
