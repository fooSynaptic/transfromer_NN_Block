# transfromer_NN_Block
We are doing this to implemented transformer as a neural network building block to overcome several task in NLP research, this rep follow the raw paper realization of [Attention Is All You Need](https://arxiv.org/abs/1706.03762).

This rep achieved **Several** tasks:
- The seq2seq text generation, we try to implemented transformer solve a conventional problem in NLP - Words segmentation(Chinese).
- [The NMT problem track on Chinese-English machine translation.](https://github.com/fooSynaptic/transfromer_NN_Block/tree/master/en-zh_NMT)
- [The language model encoder architecture for Text-classfication.](https://github.com/fooSynaptic/transfromer_NN_Block/tree/master/transformer_text_Classfication)
- [The sentence entailement task.](https://github.com/fooSynaptic/transfromer_NN_Block/tree/master/transformer_infersent)



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

# result of chinese sentences classfication(char-level)
```
              precision    recall  f1-score   support

           0       0.99      1.00      0.99       992
           1       1.00      0.99      0.99       980

   micro avg       0.99      0.99      0.99      1972
   macro avg       0.99      0.99      0.99      1972
weighted avg       0.99      0.99      0.99      1972

Done
```


# Ref

-  https://github.com/Kyubyong/transformer
-  [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
