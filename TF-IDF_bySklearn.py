from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

vectorizer = TfidfVectorizer()

tf_idf = vectorizer.fit_transform(docs)

q = input("Question:  ")

qtf_idf = vectorizer.transform([q])

answer = cosine_similarity(tf_idf,qtf_idf)

answer = answer.ravel().argsort()[-3:][::-1]

print([docs[i] for i in answer])