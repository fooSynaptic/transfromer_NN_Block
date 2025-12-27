# Natural Language Inference with Transformer

This module implements a sentence entailment inference task using the Transformer architecture on the Stanford SNLI dataset.

## Task Description

**Natural Language Inference (NLI)** determines the logical relationship between two sentences:

| Label | Description | Example |
|-------|-------------|---------|
| **Entailment** | Premise implies hypothesis | P: "A dog is running" → H: "An animal is moving" |
| **Contradiction** | Premise contradicts hypothesis | P: "A dog is running" → H: "All animals are sleeping" |
| **Neutral** | Neither entailment nor contradiction | P: "A dog is running" → H: "The dog is brown" |

## Dataset

**Stanford Natural Language Inference (SNLI)**

- Source: [Stanford SNLI](https://nlp.stanford.edu/projects/snli/)
- Size: ~570K sentence pairs
- Classes: 3 (entailment, contradiction, neutral)

## Quick Start

### 1. Download Dataset

```bash
wget https://nlp.stanford.edu/projects/snli/snli_1.0.zip
unzip snli_1.0.zip
```

### 2. Data Preparation

```bash
python data_prepare.py
python prepro.py
```

### 3. Training

```bash
python train.py
```

Training logs and checkpoints will be saved to `infersent_model_dir/`.

### 4. Evaluation

```bash
python eval.py --task infersent
```

## Results

### Training Progress

| Training Accuracy | Training Loss |
|-------------------|---------------|
| ![Accuracy](../images/infersent_train_with_SNLI_accuracy.png) | ![Loss](../images/infersent_train_SNLI_loss.png) |

### Evaluation Results

| Metric | Score |
|--------|-------|
| Accuracy | 0.76 |
| Macro Avg F1 | 0.76 |

**Detailed Classification Report:**

```
              precision    recall  f1-score   support

           0       0.82      0.76      0.79      3358
           1       0.77      0.80      0.79      3226
           2       0.70      0.73      0.72      3208

    accuracy                           0.76      9792
   macro avg       0.76      0.76      0.76      9792
weighted avg       0.76      0.76      0.76      9792
```

## Model Architecture

The model encodes both sentences using Transformer encoders and combines them for classification:

```
Sentence 1 → Encoder → Enc1
                              → Fusion → Decoder → Classification
Sentence 2 → Encoder → Enc2
```

### Key Components

- **Encoder**: Shared Transformer encoder for both sentences
- **Fusion**: Element-wise combination of sentence representations
- **Decoder**: Cross-attention between fused representation and inputs
- **Classifier**: Linear projection with L2 regularization

## File Structure

```
transformer_infersent/
├── data_load.py     # Data loading and batch generation
├── data_prepare.py  # SNLI data parsing
├── eval.py          # Model evaluation
├── hyperparams.py   # Task-specific hyperparameters
├── modules.py       # Transformer building blocks
├── prepro.py        # Vocabulary preprocessing
├── train.py         # Training script
└── README.MD        # This file
```

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [A large annotated corpus for learning natural language inference](https://nlp.stanford.edu/pubs/snli_paper.pdf)
- [Stanford SNLI Dataset](https://nlp.stanford.edu/projects/snli/)
