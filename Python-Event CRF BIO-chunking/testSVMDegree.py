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

test_sents = load3("test")
#print train_sents
#print "sent =" +str(len(train_sents))
SWN = sentlex.SWN3Lexicon()

f=open("PredictedTags.pkl", 'rb')
Eventpredicted  = pickle.load(f)
f.close()


global wordCnt 

wordCnt = -1
def word2features(sent, i):
	global wordCnt 
	wordCnt += 1
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
	Event = Eventpredicted[wordCnt]
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
		Event1 = Eventpredicted[wordCnt-1]
		
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
		Event1 = Eventpredicted[wordCnt+1]
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
		Event2 = Eventpredicted[wordCnt-2]
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
		Event2 = Eventpredicted[wordCnt+2]
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


	'''print "word : "  
	print  sent[i]
	print "features:"
	print features                
	print 
	print'''
	return features


def getNum(label):
	if(label == "LITTLE"):
		return -1
	elif(label == "MOST"):
		return 1
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
	# return [label for token, postag, norm, cui, tui, label, start, end in sent]
	return [Degree for token, postag, norm, cui, tui, label, start, end, fileName, Type, Degree, Polarity, Modality, Aspect in sent]


def sent2tokens(sent):
    # return [token for token, postag, norm, cui, tui, label, start, end in sent]    
	return [token for token, postag, norm, cui, tui, label, start, end , fileName, Type, Degree, Polarity, Modality, Aspect in sent]    

# vec = DictVectorizer()

def eventEvaluate(cor,pred):
	f=open("PredictedTags.pkl", 'rb')
	predictedEvent = pickle.load(f)
	f.close()

	f=open("CorrectTags.pkl", 'rb')
	correctEvent = pickle.load(f)
	f.close()
	ind = -1
	sysandgrnd = 0
	sys = 0
	grnd = 0
	sysandgrndAttr = 0
	for p in pred:
		ind += 1
		if(predictedEvent[ind]!="O"):
			sys += 1			
			if(correctEvent[ind]==predictedEvent[ind]):
				sysandgrnd += 1
				if(cor[ind]==pred[ind]):
					sysandgrndAttr += 1
		if(correctEvent[ind]!="O"):
			grnd += 1

	prec = sysandgrndAttr/float(sys)
	rec = sysandgrndAttr/float(grnd)
	fmes = 2 * prec * rec /(prec + rec)
	acc = sysandgrndAttr /float(sysandgrnd)
	print "Performance Measures:"
	print "Precision  = " +  str(prec)
	print "Recall  = " +  str(rec)
	print "Fmeasure  = " +  str(fmes)
	print "Accuracy = " + str(acc)


# load it again
with open('my_dumped_SVMTDegreeclassifier.pkl', 'rb') as fid:
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
		if(num==-1):
			predicted_labels.append("LITTLE")
		elif(num==1):
			predicted_labels.append("MOST")		
		else:
			predicted_labels.append("N/A")

	#print "Predict:" +str(predicted_labels)
	#print "correct : " + str(test_labels)

	wantedCorrect = []
	wantedPredicted = []
	wordCnt = -1
	for label in predicted_labels:
		wordCnt += 1
		if( test_labels[wordCnt] !="O" or predicted_labels[wordCnt]!="O" ):
			wantedCorrect.append(test_labels[wordCnt])
			wantedPredicted.append(predicted_labels[wordCnt])


	# evaluate(wantedCorrect ,wantedPredicted)
	eventEvaluate(test_labels,predicted_labels)