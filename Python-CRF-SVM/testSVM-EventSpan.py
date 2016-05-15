 import pickle
from sklearn.feature_extraction import DictVectorizer
from itertools import chain
import nltk
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import sklearn
import pycrfsuite
from loadTuples import load,load2,load3
from sklearn import svm
from evalt import *


'''test_sents = load2("test")
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



	return features'''

import sentlex
test_sents = load3("test")
# test_sents = load2("test")[:100]
#print train_sents
#print "sent =" +str(len(train_sents))
SWN = sentlex.SWN3Lexicon()

def word2features(sent, i):
	"""get the feautes corresponding to a word in a sentence at a particular position
    Args:
        sent: the sentence whose word is to be considered
        i: the position of the word in the sentence
    Returns:
        the dictionary containing the features for the classifier
    """
	word = sent[i][0]
	postag = sent[i][1]
	norm = sent[i][2]
	cui = sent[i][3]
	tui = sent[i][4]
	(pos,neg) = (0,0)
	(p1,n1) = SWN.getadjective(word)
	(p2,n2) = SWN.getnoun(word)
	(p3,n3) = SWN.getverb(word)
	(p4,n4) = SWN.getadverb(word)
	pos+= (p1+p2+p3+p4)
	neg+= (n1+n2+n3+n4)
	Pol = True
	if(neg>pos):
		Pol=False
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
		'Pol':	Pol,	
	}

	if(i > 0):
		word1 = sent[i-1][0]
		(pos1,neg1) = (0,0)
		(p11,n11) = SWN.getadjective(word1)
		(p21,n21) = SWN.getnoun(word1)
		(p31,n31) = SWN.getverb(word1)
		(p41,n41) = SWN.getadverb(word1)
		pos1+= (p11+p21+p31+p41)
		neg1+= (n11+n21+n31+n41)
		Pol1 = True
		if(neg1>pos1):
			Pol1=False
		
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
			'-1:Pol':	Pol1,	
		})
	else:
		features.update({'BOS':True})






	if i < len(sent)-1:
		word1 = sent[i+1][0]		
		(pos1,neg1) = (0,0)
		(p11,n11) = SWN.getadjective(word1)
		(p21,n21) = SWN.getnoun(word1)
		(p31,n31) = SWN.getverb(word1)
		(p41,n41) = SWN.getadverb(word1)
		pos1+= (p11+p21+p31+p41)
		neg1+= (n11+n21+n31+n41)
		Pol1 = True
		if(neg1>pos1):
			Pol1=False
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
			'+1:Pol':	Pol1,	
		})
	else:
		features.update({'EOS':True})

	if i > 1:
		word2 = sent[i-2][0]
		(pos2,neg2) = (0,0)
		(p12,n12) = SWN.getadjective(word2)
		(p22,n22) = SWN.getnoun(word2)
		(p32,n32) = SWN.getverb(word2)
		(p42,n42) = SWN.getadverb(word2)
		pos2+= (p12+p22+p32+p42)
		neg2+= (n12+n22+n32+n42)
		Pol2 = True
		if(neg2>pos2):
			Pol2=False
		postag2 = sent[i-2][1]
		norm2 = sent[i-2][2]
		cui2 = sent[i-2][3]
		tui2 = sent[i-2][4]
		features.update({
			'-2:word': word2,
			'-2:word.isupper': word2.isupper(),
			'-2:postag': postag2,
			'-2:norm': norm2,
			'-2:cui': cui2,
			'-2:tui': tui2,
			'-2:Pol':	Pol2,	
		})
	else:
		features.update({'BOS2':True})

	if i < len(sent)-2:
		word2 = sent[i+2][0]
		(pos2,neg2) = (0,0)
		(p12,n12) = SWN.getadjective(word2)
		(p22,n22) = SWN.getnoun(word2)
		(p32,n32) = SWN.getverb(word2)
		(p42,n42) = SWN.getadverb(word2)
		pos2+= (p12+p22+p32+p42)
		neg2+= (n12+n22+n32+n42)
		Pol2 = True
		if(neg2>pos2):
			Pol2=False
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
			'+2:Pol':	Pol2,	
		})
	else:
		features.update({'EOS2':True})

	
	return features

def getNum(label):
	"""get a unique number corresponding to each label

    Args:
        label: the label correposnding to which a number is to be alloted
    Returns:
        a unique number corresponding to each label

    """
	if(label == "B-EVENT"):
		return 1
	elif(label == "I-EVENT"):
		return 2
	else:
	 return 0

def sent2features(sent):
	"""get feauture vector for the sentence

    Args:
        sent: the sentence correposnding to which feauture vector is to be extracted
    Returns:
        feature vector for a sentence
    """
	feature = [word2features(sent, i) for i in range(len(sent)) ]
	
	'''print "feature for sentence" + str(sent)
	print
	print "Feature"
	print str(feature)'''
	return feature

def sent2labels(sent):
	"""get a vector of labels for the sentence

    Args:
        sent: the sentence correposnding to which label vector is to be extracted
    Returns:
        a vector of labels for the sentence

    """
	#print sent
	# return [label for token, postag, norm, cui, tui, label, start, end in sent]
	return [label for token, postag, norm, cui, tui, label, start, end,  fileName, Type, Degree, Polarity, Modality, Aspect in sent]


def sent2tokens(sent):
	"""get a vector of tokens for the sentence

    Args:
        sent: the sentence correposnding to which tokens vector is to be extracted
    Returns:
        a vector of tokens for the sentence

    """
    # return [token for token, postag, norm, cui, tui, label, start, end in sent]    
	return [token for token, postag, norm, cui, tui, label, start, end , fileName, Type, Degree, Polarity, Modality, Aspect in sent]    

# vec = DictVectorizer()


def eventEvaluate(cor,pred):
	"""Evaluates using partial matching

    Args:
        cor: list of the correct label
        pred: list of the predicted label    

    """
	ind = -1
	sysandgrnd = 0
	sys = 0
	grnd = 0
	for p in pred:
		ind += 1
		if(pred[ind]=="B-EVENT"):
			sys += 1
			if(cor[ind]==pred[ind]):
				sysandgrnd += 1
		if(cor[ind]=="B-EVENT"):
			grnd += 1

	prec = sysandgrnd/float(sys)
	rec = sysandgrnd/float(grnd)
	fmes = 2 * prec * rec /(prec + rec)
	print "Performance Measures:"
	print "Precision  = " +  str(prec)
	print "Recall  = " +  str(rec)
	print "Fmeasure  = " +  str(fmes)

def exactEvaluate(cor,pred):
	"""Evaluates using exact matching

    Args:
        cor: list of the correct label
        pred: list of the predicted label    

    """
	ind = -1
	sysandgrnd = 0
	sys = 0
	grnd = 0
	for p in pred:
		ind += 1
		if(pred[ind]=="B-EVENT"):
			sys += 1
			if(cor[ind]=="B-EVENT"):
				diff = 1
				correct = True
				while(ind+diff<len(pred) and pred[ind+diff]=="I-EVENT"):
					if(pred[ind+diff]==cor[ind+diff]):
						diff += 1
					else:
						correct = False
						break
				if(correct):
					sysandgrnd += 1

		
		if(cor[ind]=="B-EVENT"):
			grnd += 1	

	prec = sysandgrnd/float(sys)
	rec = sysandgrnd/float(grnd)
	fmes = 2 * prec * rec /(prec + rec)
	print "Performance Measures:"
	print "Precision  = " +  str(prec)
	print "Recall  = " +  str(rec)
	print "Fmeasure  = " +  str(fmes)


# load it again
with open('my_dumped_SVMclassifierNEG.pkl', 'rb') as fid:
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

	f=open("PredictedTagsEventSVM.pkl", 'wb')
	pickle.dump(predicted_labels, f)
	f.close()

	f=open("CorrectTagsEventSVM.pkl", 'wb')
	pickle.dump(test_labels, f)
	f.close()

	# print "Predict:" +str(predicted_labels)
	# print "correct : " + str(test_labels)
	# evaluate(test_labels ,predicted_labels)
	# eventEvaluate(test_labels,predicted_labels)
	exactEvaluate(test_labels,predicted_labels)





