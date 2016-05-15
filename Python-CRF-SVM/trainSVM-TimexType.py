from sklearn.feature_extraction import DictVectorizer
from itertools import chain
import nltk
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import sklearn
import pycrfsuite
from loadTuples import *
from sklearn import svm
from evalt3 import *
import pickle
import sentlex
train_sents = load6("train")
# test_sents = load2("test")[:100]
#print train_sents
#print "sent =" +str(len(train_sents))
SWN = sentlex.SWN3Lexicon()

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
	Timex = sent[i][7]
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
		Timex = sent[i-1][7]
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
		Timex = sent[i+1][7]
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
		Timex = sent[i-2][7]
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
		Timex = sent[i+2][7]
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
		Timex = sent[i-3][7]
		if(neg2>pos2):
			Pol2=False
		postag2 = sent[i-2][1]

		features.update({
			'-3:word': word2,
			'-3:word[-3:]' : word2[-3:],
			'-3:word[-2:]' : word2[-2:],
			# '-3:word.isupper': word2.isupper(),
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
		Timex = sent[i+3][7]
		if(neg2>pos2):
			Pol2=False
		postag2 = sent[i+3][1]

		features.update({
			'+3:word': word2,
			'+3:word[-3:]' : word2[-3:],
			'+3:word[-2:]' : word2[-2:],
			# '+3:word.isupper': word2.isupper(),
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

def getNum(label):
	"""get a unique number corresponding to each label

    Args:
        label: the label correposnding to which a number is to be alloted
    Returns:
        a unique number corresponding to each label

    """
	if(label == "PREPOSTEXP"):
		return 0
	elif(label == "TIME"):
		return 1
	elif(label == "DURATION"):
		return 2
	elif(label == "SET"):
		return 3
	elif(label == "QUANTIFIER"):
		return 4
	else:
	 return 5

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
	return [getNum(Class) for token, postag, label, start, end, fileName, medlabel, Class, MedClass  in sent]


def sent2tokens(sent):
	"""get a vector of tokens for the sentence

    Args:
        sent: the sentence correposnding to which tokens vector is to be extracted
    Returns:
        a vector of tokens for the sentence

    """
    # return [token for token, postag, norm, cui, tui, label, start, end in sent]    
	return [token for token, postag, label, start, end, fileName, medlabel, Class, MedClass in sent]    

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
with open('my_dumped_SVMTimexTypeclassifier.pkl', 'wb') as fid:
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
