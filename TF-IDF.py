import numpy as np 
from collections import Counter
import itertools

#from visual import show_tfidf

docs = [
    "it is a good day, I like to stay here",
    "I am happy to be here",
    "I am bob",
    "it is sunny today",
    "I have a party today",
    "it is a dog and that is a cat",
    "there are dog and cat on the tree",
    "I study hard this morning",
    "today is a good day",
    "tomorrow will be a good day",
    "I like coffee, I like book and I like apple",
    "I do not like it",
    "I am kitty, I like bob",
    "I do not care who like bob, but I like kitty",
    "It is coffee time, bring your cup",
]

docs_words = [d.replace(",","").split(" ")for d in docs]	#分词
vocab = set(itertools.chain(*docs_words))					#去重
vandi = {v:i for i,v in enumerate(vocab)}					#编号
iandv = {i:v for v,i in vandi.items()}

tf_methods = {
	"log": lambda x :np.log(1+x),
	"augmented": lambda x: 0.5 + 0.5 * x / np.max(x, axis=1, keepdims=True),
	"boolean": lambda x: np.minimum(x, 1),
	"log_avg": lambda x: (1 + safe_log(x)) / (1 + safe_log(np.mean(x, axis=1, keepdims=True))),
}

idf_methods = {
	"log":lambda x:1+np.log(len(docs)/(x+1)),
	"prob":lambda x:np.maximum(0, np.log((len(docs) - x) / (x+1))),
	"len-norm":lambda x:x/(np.sum(np.square(x)+1)),
}


def get_tf(method = "log"):
	_tf = np.zeros((len(vocab),len(docs)),dtype = np.float64)
	for i,d in enumerate(docs_words):
		print(i)
		print(d)


def get_idf(method = "log"):
	df = np.zeros((len(iandv),1))
	for i in range(len(iandv)):
		d_count = 0
		for d in docs_words:
			d_count += 1 if iandv[i] in d else 0
		df[i, 0] = d_count


	idf_fn = idf_methods.get(method,None)
	if idf_fn is None:
		raise ValueError
	return idf_fn(df)

get_tf()