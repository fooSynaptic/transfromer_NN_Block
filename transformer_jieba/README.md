# Chinese Word Segmentation with Transformer

This module implements a Chinese Word Segmentation system using the Transformer architecture for sequence labeling.

## Task Description

Chinese word segmentation is a fundamental NLP task where continuous character sequences are split into meaningful words. This is challenging because Chinese text has no explicit word boundaries (spaces).

### Tagging Scheme

We use the **B/E/S/M** tagging scheme:

| Tag | Meaning | Example |
|-----|---------|---------|
| **B** | Begin of word | 中[B] in "中国" |
| **E** | End of word | 国[E] in "中国" |
| **S** | Single character word | 的[S] |
| **M** | Middle of word | 民[M] in "人民币" |

**Example:**
```
Input:  我 爱 中 国
Tags:   S  S  B  E
Output: 我 / 爱 / 中国
```

## Dataset

The training data is located in `./dataset/`:

- `train.tags.en-zh.en` - Source text
- `train.tags.zh-en.zh` - Target text (with segmentation labels)
- `test.tags.en-zh.en` - Test source
- `test.tags.zh-en.zh` - Test target

## Quick Start

### 1. Data Preparation

```bash
# Preprocess the data
python prepro.py
```

This creates vocabulary files in the `./preprocessed/` directory.

### 2. Training

```bash
python train.py
```

Training logs and checkpoints will be saved to `seq2seq_model_dir/`.

### 3. Evaluation

```bash
python eval.py
```

## Results

| Metric | Score |
|--------|-------|
| BLEU | ~80 |

## Model Architecture

The model treats word segmentation as a sequence-to-sequence problem:

```
Input Characters → Transformer Encoder → Transformer Decoder → B/E/S/M Tags
```

### Key Features

- **Encoder**: Encodes input character sequence
- **Decoder**: Generates corresponding segmentation tags
- **Attention**: Both self-attention and cross-attention mechanisms

## File Structure

```
transformer_jieba/
├── dataset/
│   ├── train.tags.en-zh.en    # Training source
│   ├── train.tags.zh-en.zh    # Training target
│   ├── test.tags.en-zh.en     # Test source
│   ├── test.tags.zh-en.zh     # Test target
│   └── train.txt              # Raw training data
├── data_load.py               # Data loading utilities
├── data_pre.py                # Data preprocessing
├── eval.py                    # Model evaluation
├── modules.py                 # Transformer building blocks
├── prepro.py                  # Vocabulary preprocessing
├── train.py                   # Training script
└── README.md                  # This file
```

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Max Sequence Length | 100 |
| Hidden Units | 512 |
| Encoder/Decoder Blocks | 5 |
| Attention Heads | 8 |
| Dropout Rate | 0.1 |
| Learning Rate | 0.0001 |
| Batch Size | 32 |

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Chinese word segmentation overview: [ACL Anthology](https://aclanthology.org/W03-1728/)

