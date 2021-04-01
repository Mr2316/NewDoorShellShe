import numpy as np 
from collections import Counter
import itertools

np.set_printoptions(threshold=np.inf)

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

np.savetxt("tf_idf.txt",tf_idf,fmt="%f", delimiter=",")