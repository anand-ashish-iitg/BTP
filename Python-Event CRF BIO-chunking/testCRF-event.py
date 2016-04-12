from itertools import chain
import nltk
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import sklearn
import pycrfsuite
from loadTuples import load
from evalt import *
import pickle
from evalt3 import *

#print nltk.corpus.conll2002.fileids()

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

    
                
    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
	#print sent
	return [label for token, postag, norm, cui, tui, label, start, end in sent]
	#return [label for token, postag, norm, cui, tui, label in sent]

def sent2tokens(sent):
    return [token for token, postag, norm, cui, tui, label, start, end in sent]    
    #return [token for token, postag, norm, cui, tui, label in sent]    

#print sent2features(train_sents[0])[0]    


print "Doing for test"
X_test = [sent2features(s) for s in test_sents]
y_test = [sent2labels(s) for s in test_sents]

predicted = []
correct = []

tagger = pycrfsuite.Tagger()
tagger.open('tempeval2016-event.crfsuite')



for sent in test_sents:
	predicted.extend(tagger.tag(sent2features(sent)))
	correct.extend(sent2labels(sent))

f=open("PredictedTags.pkl", 'wb')
pickle.dump(predicted, f)
f.close()

f=open("CorrectTags.pkl", 'wb')
pickle.dump(correct, f)
f.close()

evaluate3(correct,predicted)
# example_sent = test_sents[4]
# print(' '.join(sent2tokens(example_sent)))

# #print("Predicted:", ' '.join(tagger.tag(sent2features(example_sent))))
# #print("Correct:  ", ' '.join(sent2labels(example_sent)))
# predicted = tagger.tag(sent2features(example_sent))
# correct = sent2labels(example_sent)
# print "\tToken\t\tPredicted\t\tCorrect"
# for i in range(0,len(predicted)):
# 	print "\t"+example_sent[i][0]+"\t\t" + predicted[i]+ "\t\t" + correct[i]