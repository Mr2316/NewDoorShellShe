#coding=utf8

import numpy as np 
from collections import Counter
import itertools

np.set_printoptions(threshold=np.inf)
np.seterr(divide='ignore',invalid='ignore')

answers = [
	"君不见 黄河之水天上来 奔流到海不复回",
	"君不见 高堂明镜悲白发 朝如青丝暮成雪",
	"人生得意须尽欢 莫使金樽空对月",
	"天生我材必有用 千金散尽还复来",
	"烹羊宰牛且为乐 会须一饮三百杯",
	"岑夫子 丹丘生 将进酒 杯莫停",
	"与君歌一曲 请君为我倾耳听",
	"钟鼓馔玉不足贵 但愿长醉不复醒",
	"古来圣贤皆寂寞 惟有饮者留其名",
	"陈王昔时宴平乐 斗酒十千恣欢谑",
	"主人何为言少钱 径须沽取对君酌",
	"五花马 千金裘 呼儿将出换美酒 与尔同销万古愁"
]

words = [list(i.replace(" ","")) for i in answers]

len_answers=len(answers)

allwords = set(itertools.chain(*words))

len_allwords = len(allwords)

vtoi = {v:i for i,v in enumerate(allwords)}
itov = {i:v for v,i in vtoi.items()}


def get_tf():
	_tf = np.zeros((len_allwords,len_answers),dtype = np.float64)
	for i,d in enumerate(words):
		counter = Counter(d)
		for v in counter.keys():
			_tf[vtoi[v],i] = counter[v] / counter.most_common(1)[0][1]

	return np.log(1+_tf)

def get_idf():
	df = np.zeros((len_allwords,1))
	for i in range(len_allwords):
		dcount = 0
		for d in words:
			dcount += 1 if itov[i] in d else 0
		df [i,0] = dcount

	return 1+np.log(len_answers/(df+1))

tf = get_tf()
idf = get_idf()

tf_idf = tf * idf

def cosine_similarity(q_vector,_tf_idf):
	vecq = q_vector/np.sqrt(np.sum(np.square(q_vector),axis=0, keepdims=True))					#[n_vcab,1]
	vecall = _tf_idf/np.sqrt(np.sum(np.square(_tf_idf),axis=0,keepdims=True))					#[n_vcab,n_docs]
	product = vecall.T.dot(vecq).ravel()

	return product

def answerSequence(q):
	unkown_v = 0																			#the number of new vcab
	v2i = vtoi.copy()
	i2v = itov.copy()
	for v in set(q):
		if v not in v2i:
			v2i[v] = len(v2i)
			i2v[len(v2i)-1] = v
			unkown_v += 1
	if unkown_v:
		_idf = np.concatenate((idf,np.zeros((unkown_v,1),dtype = np.float64)),axis =0)
		_tf_idf = np.concatenate((tf_idf,np.zeros((unkown_v,tf_idf.shape[1]),dtype = np.float64)),axis = 0)
	else:
		_idf,_tf_idf = idf,tf_idf

	counter = Counter(q)
	q_tf = np.zeros((len(_idf),1),dtype = np.float64)
	for v in counter.keys():
		q_tf[v2i[v],0] = counter[v]

	q_vector = q_tf * _idf																	#[n_vcab,1]
	answerSequence = cosine_similarity(q_vector,_tf_idf)
	return answerSequence



while  True:
	q = input("输入问题:  ")
	if q == "quit":
		break
	else:
		q = list(q)
		answer = answerSequence(q)
		top = answer.argsort()[-1:]
		print(answers[top[0]])