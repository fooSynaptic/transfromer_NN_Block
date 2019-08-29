# -*- coding: utf-8 -*-
#/usr/bin/python3

#data preparation
import codecs
import os
import argparse
import jieba
from hyperparams import feature_Block_Hyperparams as hp
from sklearn.externals import joblib
import re

def jieba_data_pre():
	with codecs.open('./dataset/train.txt', 'r', encoding = 'utf-8') as f:
		vocabset = f.readlines()
		vocabset = [x.strip() for x in vocabset]
		#print(vocabset[:10])
	zh = ''
	en = ''

	for pair in vocabset:
		try:
			z, e = pair.strip().split('\t')
			zh += z + ' '
			en += e + ' '
		except:
			zh += '<eos>'
			en += '<eos>'


	zh_sent = zh.split('<eos>')
	en_sent = en.split('<eos>')
	assert len(zh_sent) == len(en_sent), 'length of source and target not comliable'

	files = []
	for root, dirs, file in os.walk(".", topdown=False):
		files.append(file)
	#print(files)

	if 'train.tags.zh-en.zh' not in files:
		with codecs.open('./dataset/train.tags.src-tgt.src', 'w', 'utf-8') as f:
			for i in zh_sent[:int(0.8*len(zh_sent))]:
				f.write(i+'\n')

		with codecs.open('./dataset/train.tags.tgt-src.tgt', 'w', 'utf-8') as f:
			for i in en_sent[:int(0.8*len(en_sent))]:
				f.write(i+'\n')

		with codecs.open('./dataset/test.tags.src-tgt.src', 'w', 'utf-8') as f:
			for i in zh_sent[int(0.8*len(zh_sent)):]:
				f.write(i+'\n')

		with codecs.open('./dataset/test.tags.tgt-src.tgt', 'w', 'utf-8') as f:
			for i in en_sent[int(0.8*len(en_sent)):]:
				f.write(i+'\n')



def text_sum_pre():
	cnt_title_path = './dataset/content-title.txt'
	cnt_title_pair = [x.strip().split() for x in open(cnt_title_path).readlines()]
	cnt_title_pair = [x for x in cnt_title_pair if len(x) == 2]

	content_set, sum_set = [x[0] for x in cnt_title_pair], [x[1] for x in cnt_title_pair]
	pad = ['<PAD>', '<UNK>', "<S>", "</S>"]
	content_vocabs, title_vocabs = {}, {}

	for x in content_set:
		vocabs = jieba.cut(x)
		for x in vocabs:
			if x not in content_vocabs:
				content_vocabs[x] = 1
			else:
				content_vocabs[x] += 1

	for x in sum_set:
		vocabs = jieba.cut(x)
		for x in vocabs:
			if x not in title_vocabs:
				title_vocabs[x] = 1
			else:
				title_vocabs[x] += 1

	#save vocab

	if not 'textSummary' in os.listdir('./preprocessed'):
		os.mkdir('./preprocessed/textSummary')
	with codecs.open('./preprocessed/textSummary/src.vocab.tsv', 'w', 'utf-8') as f:
		for token in pad:
			f.write(token + '\t' + '1000000000' + '\n')
		for token, val in content_vocabs.items():
			f.write(token + '\t' + str(content_vocabs[token])  + '\n')
	
	with codecs.open('./preprocessed/textSummary/tgt.vocab.tsv', 'w', 'utf-8') as f:
		for token in pad:
			f.write(token + '\t' + '1000000000' + '\n')
		for token, val in title_vocabs.items():
			f.write(token + '\t' + str(title_vocabs[token])  + '\n')


	if 'textSummary' in os.listdir('./dataset'):
		os._exit(0)
	else:
		os.mkdir('./dataset/textSummary')
	
	n = len(sum_set)
	with codecs.open('./dataset/textSummary/train.tags.src-tgt.src', 'w', 'utf-8') as f:
		for x in content_set[:int(0.8*n)]:
			f.write(x+'\n')

	with codecs.open('./dataset/textSummary/train.tags.tgt-src.tgt', 'w', 'utf-8') as f:
		for x in sum_set[:int(0.8*n)]:
			f.write(x+'\n')
	
	with codecs.open('./dataset/textSummary/test.tags.src-tgt.src', 'w', 'utf-8') as f:
		for x in content_set[int(0.8*n):]:
			f.write(x+'\n')
	
	with codecs.open('./dataset/textSummary/test.tags.tgt-src.tgt', 'w', 'utf-8') as f:
		for x in sum_set[int(0.8*n):]:
			f.write(x+'\n')


def main():
	parser = argparse.ArgumentParser(description='Choice the task you want to run.')
	parser.add_argument('--task', default = 'jieba',
						help='task name(default: tokenize)')

	args = parser.parse_args()
	task_name = args.task
	
	if task_name == 'jieba': jieba_data_pre()
	elif task_name == 'textsum': text_sum_pre()


if __name__ == '__main__':
	import pandas as pd
	import random
	adspath = ['./datasets/ADs_detection.csv', './datasets/20190723_ads_annotation.csv']
	
	df = pd.read_csv(adspath.pop())
	while adspath:
		df = pd.concat([df, pd.read_csv(adspath.pop())], axis = 0)

	ads = [re.sub("[\s\p']", "", x)+'\t'+'1' for x in df['text']]
	asr = [x.strip() for x in joblib.load('./datasets/corpus.json')['content'] if len(x)>3]
	no_ads = random.sample(asr, min(len(ads)*1, len(asr)))
	no_ads = [re.sub("[\s\p']", "", x)+'\t'+'0' for x in no_ads]

	corpus = ads[:]
	corpus.extend(no_ads)

	random.shuffle(corpus)
	sep = int(0.9*len(corpus))
	with open('./datasets/trainset.txt', 'w') as f1:
		for line in corpus[:sep]:
			f1.write(line+'\n')

	with open('./datasets/testset.txt', 'w') as f2:
		for line in corpus[sep:]:
			f2.write(line+'\n')

	print('Done!')



