import numpy as np 
from collections import Counter
import itertools

np.set_printoptions(threshold=np.inf)
np.seterr(divide='ignore',invalid='ignore')

docs = [
	"Why so serious",
	"Because I am batman",
	"Things always get worse before they get better",
	"Let us put a smile on that face",
	"Now I see the funny side,now I am always smiling",
	"Sometimes people deserve to have their faith rewarded",
	"I am whatever Gotham needs me to be",
	"I am the chaos",
	"it is not who I am underneath,but what I do that defines me",
	"You either die a hero or you live long enough to see yourself become the villain",
	"I just wanna see her simle again",
	"You know,madness is a lot like gravity,sometimes all you need is a little push",
	"I belive,whatever does not kill you simply makes you stranger"
]

words_in_docs = [d.replace(","," ").split(" ") for d in docs]									#分词    [["Why","so","serious"]]

allwords = set(itertools.chain(*words_in_docs))													#拆分并去重   ["Why","so"]


vtoi = {v:i for i,v in enumerate(allwords)}														#{词:序号}
itov = {i:v for v,i in vtoi.items()}															#{序号：词}

def get_tf():
	_tf = np.zeros((len(allwords),len(docs)),dtype = np.float64)								#shape = (词数,文章数)
	for i,d in enumerate(words_in_docs):
		counter = Counter(d)
		for v in counter.keys():
			_tf[vtoi[v],i] = counter[v] / counter.most_common(1)[0][1]

	return np.log(1+_tf)


def get_idf():
	df = np.zeros((len(allwords),1))															#shape = (词数,1)
	for i in range(len(allwords)):
		dcount = 0																				#词频计数器
		for d in words_in_docs:
			dcount += 1 if itov[i] in d else 0
		df[i,0] = dcount

	return 1+np.log(len(docs)/(df + 1))

tf = get_tf()
idf = get_idf()
tf_idf = tf*idf



def cosine_similarity(q_vector,_tf_idf):
	vecq = q_vector/np.sqrt(np.sum(np.square(q_vector),axis=0, keepdims=True))					#[n_vcab,1]
	vecall = _tf_idf/np.sqrt(np.sum(np.square(_tf_idf),axis=0,keepdims=True))					#[n_vcab,n_docs]
	product = vecall.T.dot(vecq).ravel()

	return product

def answerSequence(q):
	words_in_q = q.replace(","," ").split(" ")
	unkown_v = 0																				#the number of new vcab
	for v in set(words_in_q):
		if v not in vtoi:
			vtoi[v] = len(vtoi)
			itov[len(vtoi)-1] = v
			unkown_v += 1
	if unkown_v:
		_idf = np.concatenate((idf,np.zeros((unkown_v,1),dtype = np.float64)),axis =0)
		_tf_idf = np.concatenate((tf_idf,np.zeros((unkown_v,tf_idf.shape[1]),dtype = np.float64)),axis = 0)
	else:
		_idf,_tf_idf = idf,tf_idf

	counter = Counter(words_in_q)
	q_tf = np.zeros((len(_idf),1),dtype = np.float64)
	for v in counter.keys():
		q_tf[vtoi[v],0] = counter[v]

	q_vector = q_tf * _idf																	#[n_vcab,1]
	answerSequence = cosine_similarity(q_vector,_tf_idf)
	return answerSequence


def QandA():
	q = input("Question:  ")
	answer = answerSequence(q)
	top3 = answer.argsort()[-3:][::-1]
	print([docs[i] for i in top3])

QandA()
