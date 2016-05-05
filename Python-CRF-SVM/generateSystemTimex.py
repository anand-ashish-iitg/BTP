import pickle
from sklearn.feature_extraction import DictVectorizer
from itertools import chain
import nltk
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import sklearn
import pycrfsuite
from loadTuples import *
from sklearn import svm
from evalt import *
import sentlex
import operator

test_sents = load6("test")

f=open("PredictedTagsTimexSpan.pkl", 'rb')
Timexpredicted  = pickle.load(f)
f.close()

f=open("PredictedTagsTimexType.pkl", 'rb')
Typepredicted  = pickle.load(f)
f.close()


# token, postag, label, start, end, fileName, medlabel, Class, MedClass
global wordCnt
wordCnt = -1
currFile = ""
rootPath = "System-output/"
openingStr = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<data>\n\n<schema path=\"./\" protocol=\"file\">temporal-schema.xml</schema>\n\n<annotations>\n"
endStr = "\n</annotations>\n</data>"
tid = 0
filid = None
for i in range(len(test_sents)):
	sent = test_sents[i]
	for j in range(len(sent)):	
		wordTuple = sent[j]
		wordCnt += 1
		if(wordCnt==0):# first word in corpus
			currFile = wordTuple[5]
			filid = open(rootPath+currFile+"-System-timex.xml","w+")
			filid.write(openingStr)
			tid = 0
		if(currFile!=wordTuple[5]):
			filid.write(endStr)
			filid.close()
			currFile = wordTuple[5]
			filid = open(rootPath+currFile+"-System-timex.xml","w+")
			tid = 0
			filid.write(openingStr)
		# print "wordnum = " + str(wordCnt) + " timepred  = " + Timexpredicted[wordCnt]
		if(Timexpredicted[wordCnt]=="B-TIMEX"):
			tid += 1
			stcord = wordTuple[3]
			endcord = wordTuple[4]
			diff = 1
			typed = {"DATE":0, "TIME":0,"DURATION":0,"QUANTIFIER":0,"SET":0,"PREPOSTEXP":0}
			typed[Typepredicted[wordCnt]] += 1

			while(j+diff<len(sent) and Timexpredicted[wordCnt+diff]=="I-TIMEX"):
				endcord = sent[j+diff][4]				
				typed[Typepredicted[wordCnt+diff]] += 1
				diff += 1
			j += (diff-1)


			curStr = "\n\t<entity>\n" + \
						"\t\t<id>"+str(tid)+"@e@"+currFile +"@system</id>\n" + \
						"\t\t<span>"+str(stcord)+","+str(endcord+1)+"</span>\n" + \
						"\t\t<type>TIMEX3</type>\n" + \
						"\t\t<parentsType>TemporalEntities</parentsType>\n" + \
						"\t\t<properties>\n"+ \
							"\t\t\t<Class>"+max(typed.iteritems(), key=operator.itemgetter(1))[0]+"</Class>\n" + \
						"\t\t</properties>\n" + \
					"\t</entity>\n\n"
			filid.write(curStr)

filid.close()




