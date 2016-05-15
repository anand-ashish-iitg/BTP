from sklearn.feature_extraction import DictVectorizer
from itertools import chain
import nltk
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import sklearn
import pycrfsuite
from loadTuples import load,load2,load3
from sklearn import svm
from evalt3 import *
import pickle
import sentlex
train_sents = load3("train")
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
	Event = sent[i][5]
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
		'Event': Event,
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
		Event1 =sent[i-1][5]
		
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
			'-1:Event': Event1,
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
		Event1 = sent[i+1][5]
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
			'+1:Event': Event1,
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
		Event2 = sent[i-2][5]
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
			'-2:Event': Event2,
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
		Event2 = sent[i+2][5]
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
			'+2:Event': Event2,
		})
	else:
		features.update({'EOS2':True})
	# 3rd token before after

	if i > 2:
		word2 = sent[i-3][0]
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
		Event2 = sent[i-3][5]
		postag2 = sent[i-3][1]
		norm2 = sent[i-3][2]
		cui2 = sent[i-3][3]
		tui2 = sent[i-3][4]
		features.update({
			'-3:word': word1,
			'-3:word.isupper': word2.isupper(),
			'-3:postag': postag2,
			'-3:norm': norm2,
			'-3:cui': cui2,
			'-3:tui': tui2,
			'-3:Pol':	Pol2,	
			'-3:Event': Event2,
		})
	else:
		features.update({'BOS3':True})

	if i < len(sent)-3:
		word2 = sent[i+3][0]
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
		Event2 = sent[i+3][5]
		postag2 = sent[i+3][1]
		norm2 = sent[i+3][2]
		cui2 = sent[i+3][3]
		tui2 = sent[i+3][4]
		features.update({
			'+3:word': word2,
			'+3:word.isupper': word2.isupper(),
			'+3:postag': postag2,
			'+3:norm': norm2,
			'+3:cui': cui2,
			'+3:tui': tui2,
			'+3:Pol':	Pol2,	
			'+3:Event': Event2,
		})
	else:
		features.update({'EOS3':True})

		# 4th token before after

	if i > 3:
		word2 = sent[i-4][0]
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
		Event2 = sent[i-4][5]
		postag2 = sent[i-4][1]
		norm2 = sent[i-4][2]
		cui2 = sent[i-4][3]
		tui2 = sent[i-4][4]
		features.update({
			'-4:word': word1,
			'-4:word.isupper': word2.isupper(),
			'-4:postag': postag2,
			'-4:norm': norm2,
			'-4:cui': cui2,
			'-4:tui': tui2,
			'-4:Pol':	Pol2,	
			'-4:Event': Event2,
		})
	else:
		features.update({'BOS4':True})

	if i < len(sent)-4:
		word2 = sent[i+4][0]
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
		Event2 = sent[i+4][5]
		postag2 = sent[i+4][1]
		norm2 = sent[i+4][2]
		cui2 = sent[i+4][3]
		tui2 = sent[i+4][4]
		features.update({
			'+4:word': word2,
			'+4:word.isupper': word2.isupper(),
			'+4:postag': postag2,
			'+4:norm': norm2,
			'+4:cui': cui2,
			'+4:tui': tui2,
			'+4:Pol':	Pol2,	
			'+4:Event': Event2,
		})
	else:
		features.update({'EOS4':True})

	'''print "word : "  
	print  sent[i]
	print "features:"
	print features                
	print 
	print'''
	return features

def getNum(label):
	"""get a unique number corresponding to each label

    Args:
        label: the label correposnding to which a number is to be alloted
    Returns:
        a unique number corresponding to each label

    """
	if(label == "HEDGED"):
		return 0
	elif(label == "GENERIC"):
		return 1
	elif(label == "HYPOTHETICAL"):
		return 2
	else:
	 return 3

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
	# return [getNum(label) for token, postag, norm, cui, tui, label, start, end in sent]
	return [getNum(Modality) for token, postag, norm, cui, tui, label, start, end,  fileName, Type, Degree, Polarity, Modality, Aspect in sent]


def sent2tokens(sent):
    # return [token for token, postag, norm, cui, tui, label, start, end in sent]    
	return [token for token, postag, norm, cui, tui, label, start, end , fileName, Type, Degree, Polarity, Modality, Aspect in sent]    

#print sent2features(train_sents[0])[0]    


print "Doing for train"
vec = DictVectorizer()
train_data =[]
for s in train_sents:
	train_data.extend(sent2features(s))

'''print "train_data:"
print train_data'''

print 
print
train_labels = []
for s in train_sents:
	train_labels.extend(sent2labels(s))
#print train_data
train_vectors = vec.fit_transform(train_data)
'''print "train_vectors:"
print train_vectors
print 
print'''

# print "Test part"
# test_data =[]
# for s in test_sents:
# 	test_data.extend(sent2features(s))

# test_vectors = vec.transform(test_data)

# test_labels = []
# for s in test_sents:
# 	test_labels.extend(sent2labels(s))


#classifier_rbf = svm.SVC(kernel='linear')
classifier_rbf = svm.LinearSVC()
print "Fitting"
classifier_rbf.fit(train_vectors, train_labels)
print "Dumping"


# save the classifier
with open('my_dumped_SVMModalityclassifier.pkl', 'wb') as fid:
    pickle.dump(classifier_rbf, fid)  
    pickle.dump(vec,fid)  
'''
# load it again
with open('my_dumped_classifier.pkl', 'rb') as fid:
    gnb_loaded = cPickle.load(fid)
prediction_rbf = classifier_rbf.predict(test_vectors)

prediction_rbf = list(prediction_rbf)
print "Predict:" +str(prediction_rbf)
print "correct : " + str(test_labels)
evaluate3(test_labels ,prediction_rbf)'''
