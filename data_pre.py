#data preparation
import codecs
import os


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


files = []
for root, dirs, file in os.walk(".", topdown=False):
	files.append(file)
print(files)

if 'train.tags.zh-en.zh' not in files:
	with codecs.open('./dataset/train.tags.zh-en.zh', 'w', 'utf-8') as f:
		for i in zh_sent[:int(0.8*len(zh_sent))]:
			f.write(i+'\n')

	with codecs.open('./dataset/train.tags.en-zh.en', 'w', 'utf-8') as f:
		for i in en_sent[:int(0.8*len(en_sent))]:
			f.write(i+'\n')

	with codecs.open('./dataset/test.tags.zh-en.zh', 'w', 'utf-8') as f:
		for i in zh_sent[int(0.8*len(zh_sent)):]:
			f.write(i+'\n')

	with codecs.open('./dataset/test.tags.en-zh.en', 'w', 'utf-8') as f:
		for i in en_sent[int(0.8*len(en_sent)):]:
			f.write(i+'\n')






def main():
	assert len(zh_sent) == len(en_sent)
	#print(zh_sent)
	#print(en_sent)

























'''
with codecs.open('./dataset/train.txt', 'r', encoding = 'utf-8') as f:
	vocabset = f.readlines()
	print(len(vocabset), vocabset[:10])
	zh_vocabs = [x.strip().split('\t')[0] for x in vocabset if len(x.strip()) != 0]
	en_vocabs = [y.strip().split('\t')[1] for y in vocabset if len(y.strip()) != 0]

zh_stop_vocab = ['。', '？', '！']
en_stop_vocab = ['.', '?', '!']

print('we have vocabs of {}'.format(len(zh_vocabs)))

def sent_gen(vocabs, stopset):
	sent_set = []
	start = 0
	for i in range(1, len(vocabs)-1):
		if len(i.strip()) == 0:
			sent_set.append(''.join(vocabs[start, i+1]))
	return sent_set

def main():
	print(len(sent_gen(zh_vocabs, zh_stop_vocab)),
		len(sent_gen(en_vocabs, en_stop_vocab)))
'''



if __name__ == '__main__':
	main()
