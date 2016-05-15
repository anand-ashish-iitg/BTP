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
from collections import Counter
import sentlex

test_sents = load6("test")
#print train_sents
#print "sent =" +str(len(train_sents))
SWN = sentlex.SWN3Lexicon()

f=open("PredictedTagsTimexSpan.pkl", 'rb')
Timexpredicted  = pickle.load(f)
f.close()


def getIsSpell(word):
	"""Checks whether the word is a spelling of common numbers

    Args:
        word: the word to be checked for spelling of number
    Returns:
        True if the word is spelling of common numberr
    """
	units = [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
        "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
        "sixteen", "seventeen", "eighteen", "nineteen",
      ]
	if(word.lower() in units):
		return True
	else:
		return False

def getIsQuant(word):
	"""Checks whether the word is common quantitative descriptor

    Args:
        word: the word to be checkd for quantitative descriptor
    Returns:
        True if the word is spelling of common  quantitative descriptor
    """
	quant = [
		"once", "twice","thrice","first","second","third","fourth","fifth","sixth","single","multiple",	
		]		
	if(word.lower() in quant):
		return True
	else:
		return False

def getIsPrePost(word):
	"""Checks whether the word is common pre-post expression
    Args:
        word: the word to be checkd for common pre-post expression
    Returns:
        True if the word is spelling of common pre-post expression
    """
	
	word = word.lower()
	if("pre" in word):
		return True
		
	if("post" in word):
		return True
		
	if("intra" in word):
		return True
		
	if("peri" in word):
		return True

	return False

global wordCnt 

wordCnt = -1
def word2features(sent, i):
	"""get the feautes corresponding to a word in a sentence at a particular position
    Args:
        sent: the sentence whose word is to be considered
        i: the position of the word in the sentence
    Returns:
        the dictionary containing the features for the classifier
    """
	global wordCnt 
	word = sent[i][0]
	postag = sent[i][1]
	medlabel = sent[i][6]
	medclass = sent[i][8]
	(pos,neg) = (0,0)
	(p1,n1) = SWN.getadjective(word)
	(p2,n2) = SWN.getnoun(word)
	(p3,n3) = SWN.getverb(word)
	(p4,n4) = SWN.getadverb(word)
	pos+= (p1+p2+p3+p4)
	neg+= (n1+n2+n3+n4)
	Pol = True
	Timex =Timexpredicted[wordCnt]
	if(neg>pos):
		Pol=False
	features = {
		'word=' :  word,
		'word[-3:]' : word[-3:],
		'word[-2:]' : word[-2:],
		'word[:3]' : word[:3],    
		'word[:-3]': word[:-3],    
		'word.isupper':  word.isupper(),
		'word.isdigit':  word.isdigit(),
		'postag':  postag,		
		'medlabel': medlabel,
		'Pol':	Pol,	
		'prepost':getIsPrePost(word),
		'spell':getIsSpell(word),
		'quant':getIsQuant(word),
		'medclass': medclass,
		# 'Timex':Timex,
	}

	if(i > 0):
		word1 = sent[i-1][0]
		postag1 = sent[i-1][1]
		medlabel1 = sent[i-1][6]
		medclass = sent[i-1][8]
		(pos1,neg1) = (0,0)
		(p11,n11) = SWN.getadjective(word1)
		(p21,n21) = SWN.getnoun(word1)
		(p31,n31) = SWN.getverb(word1)
		(p41,n41) = SWN.getadverb(word1)
		pos1+= (p11+p21+p31+p41)
		neg1+= (n11+n21+n31+n41)
		Pol1 = True
		Timex = Timexpredicted[wordCnt-1]
		if(neg1>pos1):
			Pol1=False
		

		features.update({
			'-1:word':  word1,
			'-1:word[-3:]' : word1[-3:],
			'-1:word[-2:]' : word1[-2:],
			'-1:word.isupper': word1.isupper(),
			'-1:word.isdigit':  word1.isdigit(),
			'-1:postag' : postag1,
			'-1:word[:-3]': word[:-3],
			'-1:medlabel': medlabel1,
			'-1:Pol':	Pol1,	
			'-1:prepost':getIsPrePost(word1),
			'-1:spell':getIsSpell(word1),
			'-1:quant':getIsQuant(word1),
			'-1:medclass': medclass,
			# '-1:Timex':Timex,
		})
	else:
		features.update({'BOS':True})






	if i < len(sent)-1:
		word1 = sent[i+1][0]	
		medlabel1 = sent[i+1][6]
		postag1 = sent[i+1][1]
		medclass = sent[i+1][8]
		(pos1,neg1) = (0,0)
		(p11,n11) = SWN.getadjective(word1)
		(p21,n21) = SWN.getnoun(word1)
		(p31,n31) = SWN.getverb(word1)
		(p41,n41) = SWN.getadverb(word1)
		pos1+= (p11+p21+p31+p41)
		neg1+= (n11+n21+n31+n41)
		Pol1 = True
		Timex = Timexpredicted[wordCnt+1]
		if(neg1>pos1):
			Pol1=False
		features.update({
			'+1:word': word1,			
			'+1:word[-3:]' : word1[-3:],
			'+1:word[-2:]' : word1[-2:],
			'+1:word.isupper': word1.isupper(),
			'+1:word.isdigit':  word1.isdigit(),
			'+1:postag': postag1,
			'+1:word[:-3]': word[:-3],
			'+1:medlabel': medlabel1,
			'+1:Pol':	Pol1,					
			'+1:prepost':getIsPrePost(word1),
			'+1:spell':getIsSpell(word1),
			'+1:quant':getIsQuant(word1),
			'+1:medclass': medclass,
			# '+1:Timex':Timex,
		})
	else:
		features.update({'EOS':True})

	if i > 1:
		word2 = sent[i-2][0]
		medlabel2 = sent[i-2][6]
		medclass = sent[i-2][8]
		(pos2,neg2) = (0,0)
		(p12,n12) = SWN.getadjective(word2)
		(p22,n22) = SWN.getnoun(word2)
		(p32,n32) = SWN.getverb(word2)
		(p42,n42) = SWN.getadverb(word2)
		pos2+= (p12+p22+p32+p42)
		neg2+= (n12+n22+n32+n42)
		Pol2 = True
		Timex = Timexpredicted[wordCnt-2]
		if(neg2>pos2):
			Pol2=False
		postag2 = sent[i-2][1]

		features.update({
			'-2:word': word2,
			'-2:word[-3:]' : word2[-3:],
			'-2:word[-2:]' : word2[-2:],
			'-2:word.isupper': word2.isupper(),
			'-2:word.isdigit':  word.isdigit(),
			'-2:postag': postag2,
			'-2:word[:-3]': word[:-3],
			'-2:medlabel':medlabel2,
			'-2:Pol':	Pol2,				
			'-2:prepost':getIsPrePost(word2),
			'-2:spell':getIsSpell(word2),
			'-2:quant':getIsQuant(word2),	
			'-2:medclass': medclass,
			# '-2:Timex':Timex,
		})
	else:
		features.update({'BOS2':True})

	if i < len(sent)-2:
		word2 = sent[i+2][0]
		medlabel2 = sent[i+2][6]
		medclass = sent[i+2][8]
		(pos2,neg2) = (0,0)
		(p12,n12) = SWN.getadjective(word2)
		(p22,n22) = SWN.getnoun(word2)
		(p32,n32) = SWN.getverb(word2)
		(p42,n42) = SWN.getadverb(word2)
		pos2+= (p12+p22+p32+p42)
		neg2+= (n12+n22+n32+n42)
		Pol2 = True
		Timex = Timexpredicted[wordCnt+2]
		if(neg2>pos2):
			Pol2=False
		postag2 = sent[i+2][1]

		features.update({
			'+2:word': word2,
			'+2:word[-3:]' : word2[-3:],
			'+2:word[-2:]' : word2[-2:],
			'+2:word.isupper': word2.isupper(),
			'+2:word.isdigit':  word.isdigit(),
			'+2:postag': postag2,
			'+2:word[:-3]': word[:-3],
			'+2:medlabel':medlabel2,
			'+2:Pol':	Pol2,		
			'+2:prepost':getIsPrePost(word2),
			'+2:spell':getIsSpell(word2),
			'+2:quant':getIsQuant(word2),
			'+2:medclass': medclass,
			# '+2:Timex':Timex,
		})
	else:
		features.update({'EOS2':True})

	# 3rd onwards

	if i > 2:
		word2 = sent[i-3][0]
		medlabel3 = sent[i-3][6]
		medclass = sent[i-3][8]
		(pos2,neg2) = (0,0)
		(p12,n12) = SWN.getadjective(word2)
		(p22,n22) = SWN.getnoun(word2)
		(p32,n32) = SWN.getverb(word2)
		(p42,n42) = SWN.getadverb(word2)
		pos2+= (p12+p22+p32+p42)
		neg2+= (n12+n22+n32+n42)
		Pol2 = True
		Timex = Timexpredicted[wordCnt-3]
		if(neg2>pos2):
			Pol2=False
		postag2 = sent[i-2][1]

		features.update({
			'-3:word': word2,
			'-3:word[-3:]' : word2[-3:],
			'-3:word[-2:]' : word2[-2:],
			'-3:word.isupper': word2.isupper(),
			'-3:word.isdigit':  word.isdigit(),
			'-3:postag': postag2,
			'-3:word[:-3]': word[:-3],
			'-3:medlabel':medlabel2,
			'-3:Pol':	Pol2,				
			'-3:prepost':getIsPrePost(word2),
			'-3:spell':getIsSpell(word2),
			'-3:quant':getIsQuant(word2),	
			'-3:medclass': medclass,
			# '-3:Timex':Timex,
		})
	else:
		features.update({'BOS3':True})

	if i < len(sent)-3:
		word2 = sent[i+3][0]
		medlabel2 = sent[i+3][6]
		medclass = sent[i+3][8]
		(pos2,neg2) = (0,0)
		(p12,n12) = SWN.getadjective(word2)
		(p22,n22) = SWN.getnoun(word2)
		(p32,n32) = SWN.getverb(word2)
		(p42,n42) = SWN.getadverb(word2)
		pos2+= (p12+p22+p32+p42)
		neg2+= (n12+n22+n32+n42)
		Pol2 = True
		Timex = Timexpredicted[wordCnt+3]
		if(neg2>pos2):
			Pol2=False
		postag2 = sent[i+3][1]

		features.update({
			'+3:word': word2,
			'+3:word[-3:]' : word2[-3:],
			'+3:word[-2:]' : word2[-2:],
			'+3:word.isupper': word2.isupper(),
			'+3:word.isdigit':  word.isdigit(),
			'+3:postag': postag2,
			'+3:word[:-3]': word[:-3],
			'+3:medlabel':medlabel2,
			'+3:Pol':	Pol2,		
			'+3:prepost':getIsPrePost(word2),
			'+3:spell':getIsSpell(word2),
			'+3:quant':getIsQuant(word2),
			'+3:medclass': medclass,
			# '+3:Timex':Timex,
		})
	else:
		features.update({'EOS3':True})
	return features




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
	return [Class for token, postag, label, start, end, fileName, medlabel, Class, MedClass  in sent]


def sent2tokens(sent):
	"""get a vector of tokens for the sentence

    Args:
        sent: the sentence correposnding to which tokens vector is to be extracted
    Returns:
        a vector of tokens for the sentence

    """
    # return [token for token, postag, norm, cui, tui, label, start, end in sent]    
	return [token for token, postag, label, start, end, fileName, medlabel, Class, MedClass  in sent]    

# vec = DictVectorizer()


# def eventEvaluate(cor,pred):
# 	f=open("PredictedTags.pkl", 'rb')
# 	predictedTimex = pickle.load(f)
# 	f.close()

# 	f=open("CorrectTags.pkl", 'rb')
# 	correctTimex = pickle.load(f)
# 	f.close()
# 	ind = -1
# 	sysandgrnd = 0
# 	sys = 0
# 	grnd = 0
# 	sysandgrndAttr = 0
# 	for p in pred:
# 		ind += 1
# 		if(predictedTimex[ind]!="O"):
# 			sys += 1			
# 			if(correctTimex[ind]==predictedTimex[ind]):
# 				sysandgrnd += 1
# 				if(cor[ind]==pred[ind]):
# 					sysandgrndAttr += 1
# 		if(correctTimex[ind]!="O"):
# 			grnd += 1

# 	prec = sysandgrndAttr/float(sys)
# 	rec = sysandgrndAttr/float(grnd)
# 	fmes = 2 * prec * rec /(prec + rec)
# 	acc = sysandgrndAttr /float(sysandgrnd)
# 	print "Performance Measures:"
# 	print "Precision  = " +  str(prec)
# 	print "Recall  = " +  str(rec)
# 	print "Fmeasure  = " +  str(fmes)
# 	# print "Accuracy = " + str(acc)
def eventEvaluate(cor,pred):
	"""Evaluates using partial matching

    Args:
        cor: list of the correct label
        pred: list of the predicted label    

    """
	f=open("PredictedTagsTimexSpan.pkl", 'rb')
	predictedTimex = pickle.load(f)
	f.close()

	f=open("CorrectTagsTimexSpan.pkl", 'rb')
	correctTimex = pickle.load(f)
	f.close()
	ind = -1
	sysandgrnd = 0
	sys = 0
	grnd = 0
	sysandgrndAttr = 0
	# print "predicted"
	# print predictedTimex
	# print "correct"
	# print correctTimex
	for p in pred:
		ind += 1		

		# if(predictedTimex[ind]!="O"):
		# 	if(pred[ind]=="I-TIMEX"):
		# 		#something
		# 		prev = ind -1
		# 		while(prev>0 and pred[prev]=="I-TIMEX"):
		# 			prev -= 1
		# 		if(prev>=0 and pred[prev]=="B-TIMEX"):
		# 			sys += 1			
		# 			if(correctTimex[ind]==predictedTimex[ind]):
		# 				sysandgrnd += 1
		# 				if(cor[ind]==pred[ind]):
		# 					sysandgrndAttr += 1
		if(predictedTimex[ind]!="O"):
			# if(pred[ind]=="I-DATE"):
			# 	prev = ind -1
			# 	while(prev>0 and pred[prev]=="I-DATE"):
			# 		prev -= 1
			# 	if(prev>=0 and pred[prev]=="B-DATE"):
			# 		sys += 1
			# 		if(cor[ind]==pred[ind]):
			# 			sysandgrnd += 1

			# elif(pred[ind]=="I-TIME"):
			# 	prev = ind -1
			# 	while(prev>0 and pred[prev]=="I-TIME"):
			# 		prev -= 1
			# 	if(prev>=0 and pred[prev]=="B-TIME"):
			# 		sys += 1
			# 		if(cor[ind]==pred[ind]):
			# 			sysandgrnd += 1

			# elif(pred[ind]=="I-DURATION"):
			# 	prev = ind -1
			# 	while(prev>0 and pred[prev]=="I-DURATION"):
			# 		prev -= 1
			# 	if(prev>=0 and pred[prev]=="B-DURATION"):
			# 		sys += 1
			# 		if(cor[ind]==pred[ind]):
			# 			sysandgrnd += 1

			# elif(pred[ind]=="I-QUANTIFIER"):
			# 	prev = ind -1
			# 	while(prev>0 and pred[prev]=="I-QUANTIFIER"):
			# 		prev -= 1
			# 	if(prev>=0 and pred[prev]=="B-QUANTIFIER"):
			# 		sys += 1
			# 		if(cor[ind]==pred[ind]):
			# 			sysandgrnd += 1
			# elif(pred[ind]=="I-PREPOSTEXP"):
			# 	prev = ind -1
			# 	while(prev>0 and pred[prev]=="I-PREPOSTEXP"):
			# 		prev -= 1
			# 	if(prev>=0 and pred[prev]=="B-PREPOSTEXP"):
			# 		sys += 1
			# 		if(cor[ind]==pred[ind]):
			# 			sysandgrnd += 1

			# elif(pred[ind]=="I-SET"):
			# 	prev = ind -1
			# 	while(prev>0 and pred[prev]=="I-SET"):
			# 		prev -= 1
			# 	if(prev>=0 and pred[prev]=="B-SET"):
			# 		sys += 1
			# 		if(cor[ind]==pred[ind]):
			# 			sysandgrnd += 1
			if(pred[ind]=="I-TIMEX"):
				prev = ind -1
				while(prev>0 and pred[prev]=="I-TIMEX"):
					prev -= 1
				if(prev>=0 and pred[prev]=="B-TIMEX"):
					sys += 1
					if(cor[ind]==pred[ind]):
						sysandgrnd += 1
			else:
				sys += 1			
				if(correctTimex[ind]==predictedTimex[ind]):
					sysandgrnd += 1
					if(cor[ind]==pred[ind]):
						sysandgrndAttr += 1

		if(correctTimex[ind]!="O"):
			grnd += 1

	prec = sysandgrndAttr/float(sys)
	rec = sysandgrndAttr/float(grnd)
	fmes = 2 * prec * rec /(prec + rec)
	acc = sysandgrndAttr /float(sysandgrnd)
	print "Performance Measures:"
	print "Precision  = " +  str(prec)
	print "Recall  = " +  str(rec)
	print "Fmeasure  = " +  str(fmes)
	# print "Accuracy = " + str(acc)


	#exact match
def exactEvaluate(cor,pred):
	"""Evaluates using exact matching

    Args:
        cor: list of the correct label
        pred: list of the predicted label    

    """
	f=open("PredictedTagsTimexSpan.pkl", 'rb')
	predictedTimex = pickle.load(f)
	f.close()

	f=open("CorrectTagsTimexSpan.pkl", 'rb')
	correctTimex = pickle.load(f)
	f.close()

	ind = -1
	sysandgrnd = 0
	sys = 0
	grnd = 0
	sysandgrndAttr = 0
	for p in pred:
		ind += 1
		if(predictedTimex[ind]=="B-TIMEX"):
			sys += 1
			if(correctTimex[ind]=="B-TIMEX"):
				diff = 1
				correct = True
				lcor = []
				lpred = []
				lcor.append(cor[ind])
				lpred.append(pred[ind])
				while(ind+diff<len(predictedTimex) and predictedTimex[ind+diff]=="I-TIMEX"):
					if(predictedTimex[ind+diff]==correctTimex[ind+diff]):
						diff += 1
						lcor.append(cor[ind+diff])
						lpred.append(pred[ind+diff])
					else:
						correct = False
						break
				if(correct):
					sysandgrnd += 1
					# if(pred[ind]==cor[ind]):
					# 	sysandgrndAttr += 1
					predval,n1 = Counter(lpred).most_common(1)[0]
					corval,n2 = Counter(lcor).most_common(1)[0]
					if(predval==corval):
						sysandgrndAttr += 1

		
		if(correctTimex[ind]=="B-TIMEX"):
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
with open('my_dumped_SVMTimexTypeclassifier.pkl', 'rb') as fid:
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
		if(num==0):
			predicted_labels.append("PREPOSTEXP")
		elif(num==1):
			predicted_labels.append("TIME")		
		elif(num==2):
			predicted_labels.append("DURATION")		
		elif(num==3):
			predicted_labels.append("SET")
		elif(num==4):
			predicted_labels.append("QUANTIFIER")
		else:
			predicted_labels.append("DATE")

	#print "Predict:" +str(predicted_labels)
	#print "correct : " + str(test_labels)

	f=open("PredictedTagsTimexType.pkl", 'wb')
	pickle.dump(predicted_labels, f)
	f.close()

	f=open("CorrectTagsTimexType.pkl", 'wb')
	pickle.dump(test_labels, f)
	f.close()	

	wantedCorrect = []
	wantedPredicted = []
	wordCnt = -1
	for label in predicted_labels:
		wordCnt += 1
		if( test_labels[wordCnt] !="O" or predicted_labels[wordCnt]!="O" ):
			wantedCorrect.append(test_labels[wordCnt])
			wantedPredicted.append(predicted_labels[wordCnt])


	# evaluate(wantedCorrect ,wantedPredicted)
	# eventEvaluate(test_labels,predicted_labels)
	exactEvaluate(test_labels,predicted_labels)