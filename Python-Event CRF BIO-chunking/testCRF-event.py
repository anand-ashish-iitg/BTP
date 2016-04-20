from itertools import chain
import nltk
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import sklearn
import pycrfsuite
from loadTuples import load,load3
from evalt import *
import pickle
from evalt import *

#print nltk.corpus.conll2002.fileids()
'''
test_sents = load("test")
#print test_sents
def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    norm = sent[i][2]
    cui = sent[i][3]
    tui = sent[i][4]
    features = [
        'bias',
        'word=' + word,
        'word[-3:]=' + word[-3:],
        'word[-2:]=' + word[-2:],
        'word[:3]=' + word[:3],        
        'word.isupper=%s' % word.isupper(),
        'word.isdigit=%s' % word.isdigit(),
        'postag=' + postag,
        'norm=' + norm,
        'cui=' + cui,
        'tui=' + tui,
    ]

    if i > 0:
		word1 = sent[i-1][0]
		postag1 = sent[i-1][1]
		norm1 = sent[i-1][2]
		cui1 = sent[i-1][3]
		tui1 = sent[i-1][4]
		features.extend([
		    '-1:word=' + word1,
		    '-1:word.isupper=%s' % word1.isupper(),
		    '-1:postag=' + postag1,
		    '-1:norm=' + norm1,
		    '-1:cui=' + cui1,
		    '-1:tui=' + tui1,
		])
    else:
        features.append('BOS')
        
    if i < len(sent)-1:
		word1 = sent[i+1][0]
		postag1 = sent[i+1][1]
		norm1 = sent[i+1][2]
		cui1 = sent[i+1][3]
		tui1 = sent[i+1][4]
		features.extend([
		    '+1:word=' + word1,
		    '+1:word.isupper=%s' % word1.isupper(),
		    '+1:postag=' + postag1,
		    '+1:norm=' + norm1,
		    '+1:cui=' + cui1,
		    '+1:tui=' + tui1,
		])
    else:
        features.append('EOS')


    if i > 1:
		word2 = sent[i-2][0]
		postag2 = sent[i-2][1]
		norm2 = sent[i-2][2]
		cui2 = sent[i-2][3]
		tui2 = sent[i-2][4]
		features.extend([
		    '-2:word=' + word1,
		    '-2:word.isupper=%s' % word2.isupper(),
		    '-2:postag=' + postag2,
		    '-2:norm=' + norm2,
		    '-2:cui=' + cui2,
		    '-2:tui=' + tui2,
		])
    else:
        features.append('BOS2')
        
    if i < len(sent)-2:
		word2 = sent[i+2][0]
		postag2 = sent[i+2][1]
		norm2 = sent[i+2][2]
		cui2 = sent[i+2][3]
		tui2 = sent[i+2][4]
		features.extend([
		    '+2:word=' + word2,
		    '+2:word.isupper=%s' % word2.isupper(),
		    '+2:postag=' + postag2,
		    '+2:norm=' + norm2,
		    '+2:cui=' + cui2,
		    '+2:tui=' + tui2,
		])
    else:
        features.append('EOS2')

    
                
    return features'''
    

import sentlex

test_sents = load3("test")
#print train_sents
#print "sent =" +str(len(train_sents))
SWN = sentlex.SWN3Lexicon()



def word2features(sent, i):
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
			'-2:word': word1,
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


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
	#print sent
	#return [label for token, postag, norm, cui, tui, label, start, end in sent]
	return [label for  token, postag, norm, cui, tui, label, start, end, fileName, Type, Degree, Polarity, Modality, Aspect in sent]

def sent2tokens(sent):
    # return [token for token, postag, norm, cui, tui, label, start, end in sent]    
    return [token for  token, postag, norm, cui, tui, label, start, end, fileName, Type, Degree, Polarity, Modality, Aspect in sent]    


#print sent2features(train_sents[0])[0]    


print "Doing for test"

predicted = []
correct = []

tagger = pycrfsuite.Tagger()
#tagger.open('tempeval2016-eventNEG.crfsuite')
tagger.open('tempeval2016-event.crfsuite')


for sent in test_sents:
	predicted.extend(tagger.tag(sent2features(sent)))
	correct.extend(sent2labels(sent))

predicted_eval = []
correct_eval = []
ind =-1
'''for pred in predicted:
	ind += 1
	if(predicted[ind]=="I-EVENT"):
		prev_ind= ind-1
		while prev_ind>=0 and predicted[prev_ind]=="I-EVENT":
			prev_ind -= 1
		if prev_ind >=0 and predicted[prev_ind] == "B-EVENT":
			predicted_eval.extend(predicted[ind])		
			correct_eval.extend(correct[ind])
		# do something
	else:
		predicted_eval.extend(predicted[ind])		
		correct_eval.extend(correct[ind])
'''



f=open("PredictedTags.pkl", 'wb')
pickle.dump(predicted, f)
f.close()

f=open("CorrectTags.pkl", 'wb')
pickle.dump(correct, f)
f.close()

def eventEvaluate(cor,pred):
	ind = -1
	sysandgrnd = 0
	sys = 0
	grnd = 0
	for p in pred:
		ind += 1
		if(pred[ind]!="O"):
			# for the Inside event tag check whether the begin tag was correctly identiied or not
			if(pred[ind]=="I-EVENT"):
				prev = ind -1
				while(prev>0 and pred[prev]=="I-EVENT"):
					prev -= 1
				if(prev>=0 and pred[prev]=="B-EVENT"):
					sys += 1
					if(cor[ind]==pred[ind]):
						sysandgrnd += 1
			else:
				sys += 1
				if(cor[ind]==pred[ind]):
					sysandgrnd += 1
		if(cor[ind]!="O"):
			grnd += 1

# def eventEvaluate(cor,pred):
# 	ind = -1
# 	sysandgrnd = 0
# 	sys = 0
# 	grnd = 0
# 	for p in pred:
# 		ind += 1
# 		if(pred[ind]!="O"):
# 			sys += 1
# 			if(cor[ind]==pred[ind]):
# 				sysandgrnd += 1
# 		if(cor[ind]!="O"):
# 			grnd += 1

# 	prec = sysandgrnd/float(sys)
# 	rec = sysandgrnd/float(grnd)
# 	fmes = 2 * prec * rec /(prec + rec)
# 	print "Performance Measures:"
# 	print "Precision  = " +  str(prec)
# 	print "Recall  = " +  str(rec)
# 	print "Fmeasure  = " +  str(fmes)			

	prec = sysandgrnd/float(sys)
	rec = sysandgrnd/float(grnd)
	fmes = 2 * prec * rec /(prec + rec)
	print "Performance Measures:"
	print "Precision  = " +  str(prec)
	print "Recall  = " +  str(rec)
	print "Fmeasure  = " +  str(fmes)



eventEvaluate(correct,predicted)




# example_sent = test_sents[4]
# print(' '.join(sent2tokens(example_sent)))

# #print("Predicted:", ' '.join(tagger.tag(sent2features(example_sent))))
# #print("Correct:  ", ' '.join(sent2labels(example_sent)))
# predicted = tagger.tag(sent2features(example_sent))
# correct = sent2labels(example_sent)
# print "\tToken\t\tPredicted\t\tCorrect"
# for i in range(0,len(predicted)):
# 	print "\t"+example_sent[i][0]+"\t\t" + predicted[i]+ "\t\t" + correct[i]