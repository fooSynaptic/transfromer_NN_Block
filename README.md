# transfromer_NN_Block
We are doing this to implemented transformer as a neural network building block to overcome several task in NLP research, this rep follow the raw paper realization of [Attention Is All You Need](https://arxiv.org/abs/1706.03762).

[![CircleCI](https://circleci.com/gh/huggingface/pytorch-transformers.svg?style=svg)](https://circleci.com/gh/fooSynaptic/transfromer_NN_Block)

This rep achieved **Several** tasks:
- [The seq2seq text generation, we try to implemented transformer solve a conventional problem in NLP - Words segmentation(Chinese).](https://github.com/fooSynaptic/transfromer_NN_Block/tree/master/transformer_jieba)
- [The NMT problem track on Chinese-English machine translation with WIT3 datasets.](https://github.com/fooSynaptic/transfromer_NN_Block/tree/master/en-zh_NMT)
- [The language model encoder architecture for Text-classfication.](https://github.com/fooSynaptic/transfromer_NN_Block/tree/master/transformer_text_Classfication)
- [The sentence entailement task experiment with stanford SNLI datasets(Natural language Inference).](https://github.com/fooSynaptic/transfromer_NN_Block/tree/master/transformer_infersent)
- [Updated reading comprehension task.](https://github.com/fooSynaptic/transfromer_NN_Block/tree/master/transformer_RC)



# INSTALL ENV:
Please run `pip install -r requirements.txt` first.


# ***First- the encoder-decoder architectures.***
# train
-The aim is train a sequence labeling model with **Transformer**. We follow the 
conventional sentence tokenize method - **/B/E/S/M** (represent the word begin/end/single word/in the middle respectively).

- We used some labeled chinese Ducuments to train my model. The raw data presented in the `./transformer_jieba/dataset` dir. Or you may want use the `./transformer_jieba/prepro.py` to preprocess the raw data.

- Just use the `python train.py` to train the model.


# eval
- Run `python eval.py`, We achieved the BLEU score nearly 80.


# ***Second - zh-en NMT***
- the train and test data was from `Web Inventory of Transcribed and Translated Talks`-**WIT3**, we train a model for English-Chinese translation model([data source](https://wit3.fbk.eu/mt.php?release=2015-01)).
- test Result:
  ![NMT result](https://github.com/fooSynaptic/transfromer_NN_Block/blob/master/images/NMT_res_BLEU.png)




# ***Third - the transformer feature extraction block***
- you may find the code in `./transformer_text_Classfication`, codes about preprocessing and training as well as evaluation locate in this path. And the wrappers usage are similar to encoder-decoder architecture.
- The chinese corpus was downloaded from [THUCTC(THU Chinese Text Classification)](http://thuctc.thunlp.org/), and we show better macro avg f1-score with over 0.05.
- ***Our model is very raw and shallow(only 8 multi-head attention projection and final linear projection) and without pre-trained embedding, you can explore performance with our code.***

# result of chinese sentences classfication(char-level)
` tagging = {'时尚':0, '教育':1, '时政':2, '体育':3, '游戏':4, '家居':5, '科技':6, '房产':7, '财经':8, '娱乐':9} `
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

Done
```
[***We Also implemented a sentences entailment inference task with transformer***](https://github.com/fooSynaptic/transfromer_NN_Block/tree/master/transformer_infersent)
---
**Data source** [standord SNLI](https://nlp.stanford.edu/projects/snli/snli_1.0.zip)

- *Download source data and unzip* : `wget https://nlp.stanford.edu/projects/snli/snli_1.0.zip && unzip snli_1.0.zip`
- *preprocess data*: `python data_prepare.py && python prepro.py`
- *train*: run `python train.py`
- *eval*: run `python eval.py --task infersent`

Experiment result:
- train accuracy:
![train accuracy](https://github.com/fooSynaptic/transfromer_NN_Block/blob/master/images/infersent_train_with_SNLI_accuracy.png)

- train loss:
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


# Ref

-  https://github.com/Kyubyong/transformer
-  [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
