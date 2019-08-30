# transfromer_NN_Block
We are doing this to implemented transformer as a neural network building block to overcome several task in NLP research, this rep follow the implementation if [Attention Is All You Need](https://arxiv.org/abs/1706.03762).

This rep fullfilled two tasks:
- The seq2seq text generation, we try to implemented transformer in a conventional problem in NLP-Words segment(Chinese).
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


# result demo
```
- source: 采 取 现 场 报 名 与 资 格 审 查 方 式 进 行 。
- expected: 采取 现场 报名 与 资格 审查 方式 进行 。
- got: 采取 现场 报 名与 资格 审查 方式 进 。

- source: 本 次 招 聘 于 2 0 1 5 年 8 月 4 日 - 8 月 1 0 日 接 受 现 场 报 名 及 资 格 审 查 。
- expected: 本次 招聘 于 2015年 8月 4日 - 8月 10日 接受 现场 报名 及 资格 审查 。
- got: 本次 招聘于 20 15 年8月4 日- 8 月1 0日 接受 现场 报 名及 资格 审查 。

- source: 每 位 报 考 者 在 报 名 与 考 试 时 使 用 的 身 份 证 必 须 一 致 。
- expected: 每位 报考者 在 报名 与 考试 时 使用 的 身份证 必须 一致 。
- got: 每位 报考者 在 报名 与 考试 时 使用 的身 份 证必须 一致 。

- source: 报 名 提 交 的 报 考 申 请 材 料 必 须 真 实 、 准 确 。
- expected: 报名 提交 的 报考 申请 材料 必须 真实 、 准确 。
- got: 报名 提交 的 报考 申请 材料 必须 真 、 准确 。

```

# ***Second - the transformer feature extraction block***
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
