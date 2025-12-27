# Reading Comprehension with Transformer

This module implements a Reading Comprehension (RC) model combining Transformer attention with BiDAF-style matching and Pointer Networks.

## Task Description

**Extractive Question Answering**: Given a passage and a question, the model identifies the answer span within the passage.

## Model Architecture

The architecture combines three powerful components:

```
┌─────────────────────────────────────────────────────────────┐
│                    Reading Comprehension Model               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Question ──→ Transformer Encoder ──→ Q_enc                 │
│                                          ↓                  │
│                                    ┌─────────────┐          │
│  Passage ──→ Transformer Decoder ──→│ BiDAF Match │──→ P*   │
│              (cross-attention)      └─────────────┘          │
│                                          ↓                  │
│                                   ┌──────────────┐          │
│                                   │Pointer Network│          │
│                                   └──────────────┘          │
│                                     ↓        ↓              │
│                                  Start      End             │
│                                  Position   Position        │
└─────────────────────────────────────────────────────────────┘
```

### Components

1. **Transformer Encoder**: Parallel self-attention for question and passage
2. **BiDAF Attention Flow**: Question-aware passage representation
3. **Pointer Network**: Predicts start and end positions of the answer

## Quick Start

### 1. Data Preparation

```bash
python prepro.py
```

### 2. Training

```bash
python train.py
```

Training logs and checkpoints will be saved to `rc_model_dir/`.

### 3. Evaluation

```bash
python eval.py
```

## Results

| Metric | Score |
|--------|-------|
| Rouge-L | 0.2651 |
| BLEU-1 | 0.36 |

### Training Loss

![Training Loss](../images/rc_model_train_loss.png)

Sample predictions can be found in [`results/rc_model_epoch_50_gs_10500`](../results/rc_model_epoch_50_gs_10500).

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Question Max Length | 50 |
| Passage Max Length | 200 |
| Answer Max Length | 40 |
| Hidden Units | 256 |
| Encoder Blocks | 5 |
| Attention Heads | 8 |
| Dropout Rate | 0.5 |
| Learning Rate | 0.0001 |
| Batch Size | 64 |

## File Structure

```
transformer_RC/
├── layers/
│   ├── basic_rnn.py      # RNN utilities
│   ├── match_layer.py    # BiDAF attention implementation
│   └── pointer_net.py    # Pointer Network decoder
├── data_load.py          # Data loading and batch generation
├── eval.py               # Model evaluation
├── hyperparams.py        # Task-specific hyperparameters
├── modules.py            # Transformer building blocks
├── prepro.py             # Data preprocessing
├── train.py              # Training script
└── README.md             # This file
```

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Bidirectional Attention Flow for Machine Comprehension](https://arxiv.org/abs/1611.01603)
- [Pointer Networks](https://arxiv.org/abs/1506.03134)

---

**Note**: Results are continuously being updated. Contributions and suggestions are welcome!
