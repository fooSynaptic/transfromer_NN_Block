***We implemented a sentences entailment inference task with transformer***
---
**Data source** [standord SNLI](https://nlp.stanford.edu/projects/snli/snli_1.0.zip)

- *Download source data and unzip* : `wget https://nlp.stanford.edu/projects/snli/snli_1.0.zip && unzip snli_1.0.zip`
- *preprocess data*: `python data_prepare.py && python prepro.py`
- *train*: run `python train.py`
- *eval*: run `python eval.py --task infersent`

Experiment result:
- accuracy:
![train accuracy](https://github.com/fooSynaptic/transfromer_NN_Block/blob/master/images/infersent_train_with_SNLI_accuracy.png)

- loss:
![train loss](https://github.com/fooSynaptic/transfromer_NN_Block/blob/master/images/infersent_train_SNLI_loss.png)

- eval result:
```
              precision    recall  f1-score   support

           0       0.82      0.76      0.79      3358
           1       0.77      0.80      0.79      3226
           2       0.70      0.73      0.72      3208

    accuracy                           0.76      9792
   macro avg       0.76      0.76      0.76      9792
weighted avg       0.76      0.76      0.76      9792
```
