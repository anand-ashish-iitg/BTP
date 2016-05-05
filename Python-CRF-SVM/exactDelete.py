import pickle
from sklearn.feature_extraction import DictVectorizer
from itertools import chain
import nltk
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import sklearn
import pycrfsuite
from loadTuples import load,load2, load6
from sklearn import svm
from evalt import *
from collections import Counter
import sentlex

test_sents = load6("test")

f=open("PredictedTagsSVM-TIMEXSpan.pkl", 'rb')
predictedEvent = pickle.load(f)
f.close()

f=open("CorrectTagsSVM-TIMEXSpan.pkl", 'rb')
correctEvent = pickle.load(f)
f.close()

f=open("deleteTest.txt", 'wb')
tokens = []
ind = -1
for sent in test_sents:
	for tup in sent:
		ind += 1
		f.write(tup[0] + " " + correctEvent[ind] +" " + predictedEvent[ind]+"\n")
	f.write("\n")
f.close()	
