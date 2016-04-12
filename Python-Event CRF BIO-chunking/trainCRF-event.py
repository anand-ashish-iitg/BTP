from itertools import chain
import nltk
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import sklearn
import pycrfsuite
from loadTuples import load

#print nltk.corpus.conll2002.fileids()

train_sents = load("train")
#print train_sents
#print "sent =" +str(len(train_sents))
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

def sent2tokens(sent):
    return [token for token, postag, norm, cui, tui, label, start, end in sent]    

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
    'max_iterations': 50,  # stop earlier

    # include transitions that are possible, but not observed
    'feature.possible_transitions': True
})

print trainer.params()
trainer.train('tempeval2016-event.crfsuite')

