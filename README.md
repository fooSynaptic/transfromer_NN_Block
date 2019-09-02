# transfromer_NN_Block
We are doing this to implemented transformer as a neural network building block to overcome several task in NLP research, this rep follow the implementation if [Attention Is All You Need](https://arxiv.org/abs/1706.03762).

This rep fullfilled two tasks:
- The seq2seq text generation, we try to implemented transformer in a conventional problem in NLP-Words segment(Chinese).
- The NMT problem track on zh-en machine translation.
- The text feature encode architecture for classfication.



# INSTALL ENV:
Please run `pip install -r requirements.txt` first.


# ***First- the encoder-decoder architecture***
# train
-The aim of This rep is train a sequence labeling model with **Transformer**. I follow the 
conventional sentence tokenize method - **/B/E/S/M** (represent the word begin/end/single word/in the middle respectively).

- I used some labeled chinese Ducuments to train my model. The raw data presented in the `./dataset` dir. Or you may want use the `prepro.py` to preprocess the raw data.

- Just use the `python train.py` to train the model.


# eval
- Run `python eval.py`.

For the labeling task the BLEU was implemented to evalate the model. Our model accieved the Bleu score nearly reach to the 80.

`Bleu Score = 79.347268988273`(See the `./result/model_epoch_20_gs_1560`)


# ***Second - zh-en NMT***
- the train and test data was from `Web Inventory of Transcribed and Translated Talks`-**WIT3**, we train a model for English-Chinese translation model.
- test Result:
  ![NMT result](https://github.com/fooSynaptic/transfromer_NN_Block/blob/master/NMT_res_BLEU.png)




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
