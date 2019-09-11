***We implemented a sentences entailment inference task with transformer***
Data source (standord SNLI)[https://nlp.stanford.edu/projects/snli/snli_1.0.zip]

- *Download source data and unzip* : `wget https://nlp.stanford.edu/projects/snli/snli_1.0.zip && unzip snli_1.0.zip`
- *preprocess data*: `python data_prepare.py && python prepro.py`
- *train*: run `python train.py`
- *eval*: run `python eval.py --task infersent`

Experiment result:
- accuracy:
![train accuracy](https://github.com/fooSynaptic/transfromer_NN_Block/blob/master/images/infersent_train_with_SNLI_accuracy.png)

- loss:
![train loss](https://github.com/fooSynaptic/transfromer_NN_Block/blob/master/images/infersent_train_SNLI_loss.png)
