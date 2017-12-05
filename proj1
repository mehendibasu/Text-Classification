import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn import svm,naive_bayes
from sklearn.cross_validation import train_test_split
from stemming.porter2 import stem
df=pd.read_csv("C:\\Users\\Mehendi\\Documents\\finaldataset.csv",encoding="ISO-8859-1")
cv=CountVectorizer(stop_words="english")
#cv1=TfidfVectorizer()
x=df["REVIEW"]
y=df["GENRE"]
#x = [" ".join([stem(word) for word in sentence.split(" ")]) for sentence in x]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=4)
x_traincv=cv.fit_transform(x_train)
#x_train=cv1.fit_transform(x_train)
x_traincv=x_traincv.toarray()
d={'HORROR':0,'ROMANCE':1,'THRILLER':2,"SCI-FI":3,"COMEDY":4}
y_train=y_train.apply(lambda x:d[x])
y_test=y_test.apply(lambda x:d[x])
#clf=svm.SVC(kernel="linear")
clf=naive_bayes.MultinomialNB(alpha=0.5)
clf.fit(x_traincv,y_train)
#x_test=[""]
x_testcv=cv.transform(x_test)
#x_testcv=cv1.transform(x_testcv)
x_testcv=x_testcv.toarray()
pred=clf.predict(x_testcv)
print(pred)
print(np.array(y_test))
print(clf.score(x_testcv,y_test))
for i in pred:
	if i==0:
		print("HORROR")
	elif i==1:
		print("ROMANCE")
	elif i==2:
		print("THRILLER")
	elif i==3:
		print("SCI_FI")
	else:
		print("COMEDY")

