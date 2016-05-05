from itertools import chain
import nltk
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import sklearn
import pycrfsuite
from loadTuples import load,load5

#print nltk.corpus.conll2002.fileids()


def getIsSpell(word):
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
	quant = [
		"once", "twice","thrice","first","second","third","fourth","fifth","sixth","single","multiple",	
		]		
	if(word.lower() in quant):
		return True
	else:
		return False

def getIsPrePost(word):
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


import sentlex
train_sents = load5("train")
# test_sents = load2("test")[:100]
#print train_sents
#print "sent =" +str(len(train_sents))
SWN = sentlex.SWN3Lexicon()

def word2features(sent, i):
	word = sent[i][0]
	postag = sent[i][1]
	medlabel = sent[i][6]
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
		'word[:-3]': word[:-3],    
		'word.isupper':  word.isupper(),
		'word.isdigit':  word.isdigit(),
		'postag':  postag,		
		'medlabel': medlabel,
		'Pol':	Pol,	
		'prepost':getIsPrePost(word),
		'spell':getIsSpell(word),
		'quant':getIsQuant(word),
	}

	if(i > 0):
		word1 = sent[i-1][0]
		postag1 = sent[i-1][1]
		medlabel1 = sent[i-1][6]
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
		})
	else:
		features.update({'BOS':True})






	if i < len(sent)-1:
		word1 = sent[i+1][0]	
		medlabel1 = sent[i+1][6]
		postag1 = sent[i+1][1]

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
		})
	else:
		features.update({'EOS':True})

	if i > 1:
		word2 = sent[i-2][0]
		medlabel2 = sent[i-2][6]

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
		})
	else:
		features.update({'BOS2':True})

	if i < len(sent)-2:
		word2 = sent[i+2][0]
		medlabel2 = sent[i+2][6]

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
		})
	else:
		features.update({'EOS2':True})

	# 3rd onwards

	if i > 2:
		word2 = sent[i-3][0]
		medlabel3 = sent[i-3][6]

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
		})
	else:
		features.update({'BOS3':True})

	if i < len(sent)-3:
		word2 = sent[i+3][0]
		medlabel2 = sent[i+3][6]

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
		})
	else:
		features.update({'EOS3':True})
	return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
	#print sent
	#return [label for token, postag, norm, cui, tui, label, start, end in sent]
	return [label for token, postag, label, start, end, fileName, medlabel, Class in sent]

def sent2tokens(sent):
    #return [token for token, postag, norm, cui, tui, label, start, end  in sent]    
    return [token for token, postag, label, start, end, fileName, medlabel, Class in sent]
#print sent2features(train_sents[0])[0]    

print "Doing for train"
X_train = [sent2features(s) for s in train_sents]
y_train = [sent2labels(s) for s in train_sents]



trainer = pycrfsuite.Trainer(verbose=False)

for xseq, yseq in zip(X_train, y_train):
    trainer.append(xseq, yseq)

trainer.set_params({
    'c1': 1.0,   # coefficient for L1 penalty
    'c2': 1e-3,  # coefficient for L2 penalty
    'max_iterations': 75,  # stop earlier

    # include transitions that are possible, but not observed
    'feature.possible_transitions': True
})

print trainer.params()
trainer.train('tempeval2016-timex.crfsuite')

