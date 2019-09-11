# encoding = utf-8
# /usr/bin/python3

import json

#{'entailment', '-', 'contradiction', 'neutral'}
hashmap = {'entailment':'0', 'contradiction':'1', 'neutral':'2'}


def prepare():
	train, dev, test = [[json.loads(line) for line in open('./snli_1.0/snli_1.0_{}.jsonl'.format(x)).readlines()]\
	 for x in ['train', 'dev', 'test']]

	train, dev, test = ['<>'.join([hashmap[x['gold_label']], x['sentence1'], x['sentence2']]) for x in train if x['gold_label'] in hashmap], ['<>'.join([hashmap[x['gold_label']], x['sentence1'], x['sentence2']]) for x in dev if x['gold_label'] in hashmap], ['<>'.join([hashmap[x['gold_label']], x['sentence1'], x['sentence2']]) for x in test if x['gold_label'] in hashmap]


	with open('./train.csv', 'w') as f1:
		for line in train: f1.write(line + '\n')

	with open('./dev.csv', 'w') as f2:
		for line in dev: f2.write(line + '\n')

	with open('./test.csv', 'w') as f3:
		for line in test: f3.write(line + '\n')


if __name__ == '__main__':
	prepare()

