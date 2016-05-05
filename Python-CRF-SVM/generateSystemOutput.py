import pickle
from sklearn.feature_extraction import DictVectorizer
from itertools import chain
import nltk
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import sklearn
import pycrfsuite
from loadTuples import load,load2, load3
from sklearn import svm
from evalt import *
import sentlex
import operator

test_sents = load3("test")

f=open("PredictedTags.pkl", 'rb')
Eventpredicted  = pickle.load(f)
f.close()

f=open("PredictedTagsDegree.pkl", 'rb')
Degreepredicted  = pickle.load(f)
f.close()

f=open("PredictedTagsDoctime.pkl", 'rb')
Doctimepredicted  = pickle.load(f)
f.close()

f=open("PredictedTagsModality.pkl", 'rb')
Modalitypredicted  = pickle.load(f)
f.close()

f=open("PredictedTagsPolarity.pkl", 'rb')
Polaritypredicted  = pickle.load(f)
f.close()

f=open("PredictedTagsType.pkl", 'rb')
Typepredicted  = pickle.load(f)
f.close()
global wordCnt
wordCnt = -1
currFile = ""
rootPath = "System-output/"
openingStr = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<data>\n\n<schema path=\"./\" protocol=\"file\">temporal-schema.xml</schema>\n\n<annotations>\n"
endStr = "\n</annotations>\n</data>"
evid = 0
filid = None
for i in range(len(test_sents)):
	sent = test_sents[i]
	for j in range(len(sent)):	
		wordTuple = sent[j]
		wordCnt += 1
		if(wordCnt==0):# first word in corpus
			currFile = wordTuple[8]
			filid = open(rootPath+currFile+"-System-event.xml","w+")
			filid.write(openingStr)
			evid = 0
		if(currFile!=wordTuple[8]):
			filid.write(endStr)
			filid.close()
			currFile = wordTuple[8]
			filid = open(rootPath+currFile+"-System-event.xml","w+")
			evid = 0
			filid.write(openingStr)
		if(Eventpredicted[wordCnt]=="B-EVENT"):
			evid += 1
			stcord = wordTuple[6]
			endcord = wordTuple[7]
			diff = 1
			doctime = {"BEFORE":0,"OVERLAP":0,"AFTER":0,"BEFORE/OVERLAP":0}
			typed = {"N/A":0, "ASPECTUAL":0,"EVIDENTIAL":0}
			degree = {"N/A":0, "MOST":0,"LITTLE":0}
			polarity = {"POS":0,"NEG":0}
			modality = {"ACTUAL":0, "HEDGED":0,"HYPOTHETICAL":0,"GENERIC":0}

			doctime[Doctimepredicted[wordCnt]] += 1
			typed[Typepredicted[wordCnt]] += 1
			degree[Degreepredicted[wordCnt]] += 1
			polarity[Polaritypredicted[wordCnt]] += 1
			modality[Modalitypredicted[wordCnt]] +=1

			while(j+diff<len(sent) and Eventpredicted[wordCnt+diff]=="I-EVENT"):
				endcord = sent[j+diff][7]
				doctime[Doctimepredicted[wordCnt+diff]] += 1
				typed[Typepredicted[wordCnt+diff]] += 1
				degree[Degreepredicted[wordCnt+diff]] += 1
				polarity[Polaritypredicted[wordCnt+diff]] += 1
				modality[Modalitypredicted[wordCnt+diff]] +=1
				diff += 1
			j += (diff-1)


			curStr = "\n\t<entity>\n" + \
						"\t\t<id>"+str(evid)+"@e@"+currFile +"@system</id>\n" + \
						"\t\t<span>"+str(stcord)+","+str(endcord+1)+"</span>\n" + \
						"\t\t<type>EVENT</type>\n" + \
						"\t\t<parentsType>TemporalEntities</parentsType>\n" + \
						"\t\t<properties>\n"+ \
							"\t\t\t<DocTimeRel>"+max(doctime.iteritems(), key=operator.itemgetter(1))[0]+"</DocTimeRel>\n" + \
							"\t\t\t<Type>"+max(typed.iteritems(), key=operator.itemgetter(1))[0]+"</Type>\n" + \
							"\t\t\t<Degree>"+max(degree.iteritems(), key=operator.itemgetter(1))[0]+"</Degree>\n" + \
							"\t\t\t<Polarity>"+max(polarity.iteritems(), key=operator.itemgetter(1))[0]+"</Polarity>\n" + \
							"\t\t\t<ContextualModality>"+max(modality.iteritems(), key=operator.itemgetter(1))[0]+"</ContextualModality>\n" + \
						"\t\t</properties>\n" + \
					"\t</entity>\n\n"
			filid.write(curStr)

filid.close()




