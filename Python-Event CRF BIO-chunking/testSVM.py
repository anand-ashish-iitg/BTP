import pickle
from sklearn.feature_extraction import DictVectorizer
from itertools import chain
import nltk
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import sklearn
import pycrfsuite
from loadTuples import load,load2
from sklearn import svm
from evalt import *


test_sents = load2("test")
#print train_sents
#print "sent =" +str(len(train_sents))
def word2features(sent, i):

	word = sent[i][0]
	postag = sent[i][1]
	norm = sent[i][2]
	cui = sent[i][3]
	tui = sent[i][4]
	features = {
		'word=' :  word,
		'word[-3:]' : word[-3:],
		'word[-2:]' : word[-2:],
		'word[:3]' : word[:3],        
		'word.isupper':  word.isupper(),
		'word.isdigit':  word.isdigit(),
		'postag':  postag,
		'norm':  norm,
		'cui' : cui,
		'tui:' : tui,
	}
	if(i > 0):
		word1 = sent[i-1][0]
		postag1 = sent[i-1][1]
		norm1 = sent[i-1][2]
		cui1 = sent[i-1][3]
		tui1 = sent[i-1][4]
		features.update({
			'-1:word':  word1,
			'-1:word.isupper': word1.isupper(),
			'-1:postag' : postag1,
			'-1:norm' : norm1,
			'-1:cui' : cui1,
			'-1:tui' : tui1,
		})
	else:
		features.update({'BOS':True})






	if i < len(sent)-1:
		word1 = sent[i+1][0]
		postag1 = sent[i+1][1]
		norm1 = sent[i+1][2]
		cui1 = sent[i+1][3]
		tui1 = sent[i+1][4]
		features.update({
			'+1:word': word1,
			'+1:word.isupper': word1.isupper(),
			'+1:postag': postag1,
			'+1:norm': norm1,
			'+1:cui': cui1,
			'+1:tui': tui1,
		})
	else:
		features.update({'EOS':True})

	if i > 1:
		word2 = sent[i-2][0]
		postag2 = sent[i-2][1]
		norm2 = sent[i-2][2]
		cui2 = sent[i-2][3]
		tui2 = sent[i-2][4]
		features.update({
			'-2:word': word1,
			'-2:word.isupper': word2.isupper(),
			'-2:postag': postag2,
			'-2:norm': norm2,
			'-2:cui': cui2,
			'-2:tui': tui2,
		})
	else:
		features.update({'BOS2':True})

	if i < len(sent)-2:
		word2 = sent[i+2][0]
		postag2 = sent[i+2][1]
		norm2 = sent[i+2][2]
		cui2 = sent[i+2][3]
		tui2 = sent[i+2][4]
		features.update({
			'+2:word': word2,
			'+2:word.isupper': word2.isupper(),
			'+2:postag': postag2,
			'+2:norm': norm2,
			'+2:cui': cui2,
			'+2:tui': tui2,
		})
	else:
		features.update({'EOS2':True})



	'''print "word : "  
	print  sent[i]
	print "features:"
	print features                
	print 
	print'''
	return features

def getNum(label):
	if(label == "B-EVENT"):
		return 1
	elif(label == "I-EVENT"):
		return 2
	else:
	 return 0

def sent2features(sent):
	feature = [word2features(sent, i) for i in range(len(sent)) ]
	
	'''print "feature for sentence" + str(sent)
	print
	print "Feature"
	print str(feature)'''
	return feature

def sent2labels(sent):
	#print sent
	return [label for token, postag, norm, cui, tui, label, start, end in sent]
	#return [getNum(label) for token, postag, norm, cui, tui, label, start, end, , fileName, Type, Degree, Polarity, Modality, Aspect in sent]


def sent2tokens(sent):
    return [token for token, postag, norm, cui, tui, label, start, end in sent]    
	#return [token for token, postag, norm, cui, tui, label, start, end , fileName, Type, Degree, Polarity, Modality, Aspect in sent]    

# vec = DictVectorizer()




# load it again
with open('my_dumped_SVMclassifier.pkl', 'rb') as fid:
	classifier_rbf = pickle.load(fid)
	vec =  pickle.load(fid)
	print "Test part"
	test_data =[]
	for s in test_sents:
		test_data.extend(sent2features(s))

	test_vectors = vec.transform(test_data)

	test_labels = []
	for s in test_sents:
		test_labels.extend(sent2labels(s))

	
	prediction_rbf = classifier_rbf.predict(test_vectors)	
	prediction_rbf = list(prediction_rbf)
	predicted_labels = []
	for num in  prediction_rbf:
		if(num==1):
			predicted_labels.append("B-EVENT")
		elif(num==2):
			predicted_labels.append("I-EVENT")
		else:
			predicted_labels.append("O")		

	print "Predict:" +str(predicted_labels)
	print "correct : " + str(test_labels)
	evaluate(test_labels ,predicted_labels)