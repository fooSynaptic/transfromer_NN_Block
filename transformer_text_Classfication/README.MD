# Chinese Text Classification

This module implements a Chinese text classification system using Transformer encoder as a feature extractor.

## Dataset

**THU Chinese Text Classification (THUCTC)**

- Source: [THUCTC](http://thuctc.thunlp.org/)
- 10 categories of Chinese news articles
- Character-level tokenization

### Categories

| ID | Category (Chinese) | Category (English) |
|----|-------------------|-------------------|
| 0 | 时尚 | Fashion |
| 1 | 教育 | Education |
| 2 | 时政 | Politics |
| 3 | 体育 | Sports |
| 4 | 游戏 | Gaming |
| 5 | 家居 | Home & Living |
| 6 | 科技 | Technology |
| 7 | 房产 | Real Estate |
| 8 | 财经 | Finance |
| 9 | 娱乐 | Entertainment |

## Quick Start

### 1. Data Preparation

```bash
# Preprocess the data
python prepro.py
```

### 2. Training

```bash
python train.py
```

Training logs and checkpoints will be saved to `Block_model_dir/`.

### 3. Evaluation

```bash
python eval.py
```

## Results

**10-class Classification Performance:**

| Metric | Score |
|--------|-------|
| Accuracy | 0.85 |
| Macro Avg F1 | 0.85 |
| Weighted Avg F1 | 0.85 |

<details>
<summary>Detailed Classification Report</summary>

```
              precision    recall  f1-score   support

           0       0.91      0.95      0.93      1000
           1       0.96      0.77      0.85      1000
           2       0.92      0.93      0.92      1000
           3       0.95      0.93      0.94      1000
           4       0.86      0.91      0.88      1000
           5       0.83      0.47      0.60      1000
           6       0.86      0.85      0.86      1000
           7       0.64      0.87      0.74      1000
           8       0.79      0.91      0.85      1000
           9       0.88      0.91      0.89      1000

    accuracy                           0.85     10000
   macro avg       0.86      0.85      0.85     10000
weighted avg       0.86      0.85      0.85     10000
```

</details>

## Model Architecture

The model uses a **shallow Transformer encoder** as a feature extractor:

- **Encoder Blocks**: 8 multi-head attention layers
- **Final Layer**: Linear projection for classification
- **No Pre-trained Embeddings**: Trained from scratch

### Architecture Diagram

```
Input Text → Embedding → Positional Encoding → Transformer Encoder → Pooling → Classification
```

## File Structure

```
transformer_text_Classfication/
├── data_load.py    # Data loading and batch generation
├── data_pre.py     # Data preprocessing utilities
├── eval.py         # Model evaluation
├── hyperparams.py  # Task-specific hyperparameters
├── modules.py      # Transformer building blocks
├── prepro.py       # Vocabulary and data preprocessing
├── train.py        # Training script
└── README.MD       # This file
```

## Notes

- This is a **character-level** model for Chinese text
- The model is intentionally shallow to demonstrate the effectiveness of Transformer attention
- Performance can be improved with pre-trained embeddings and deeper architectures

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [THUCTC Dataset](http://thuctc.thunlp.org/)
