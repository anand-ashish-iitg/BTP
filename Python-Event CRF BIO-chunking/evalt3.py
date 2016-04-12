from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score
from nltk.metrics import ConfusionMatrix


def evaluate3(ref,tagged):
	print '\n Confusion matrix:\n',
	cm = ConfusionMatrix(ref, tagged)
	print cm
	print
	print 'Accuracy:', accuracy_score(ref, tagged) 
	print
	print "Macro Averaged measures:"
	print 'F1 score Macro:', f1_score(ref, tagged,average='macro')
	print 'Recall Macro:', recall_score(ref, tagged, average='macro')
	print 'Precision Macro:', precision_score(ref, tagged, average='macro')
	print
	print "Micro Averaged measures:"
	print 'F1 score Micro:', f1_score(ref, tagged,average='micro')
	print 'Recall Micro:', recall_score(ref, tagged, average='micro')
	print 'Precision Micro:', precision_score(ref, tagged, average='micro')
	print
	print '\n Clasification report:\n', classification_report(ref, tagged)
	print
