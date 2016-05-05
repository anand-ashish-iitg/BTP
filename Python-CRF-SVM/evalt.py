from __future__ import print_function, unicode_literals
from nltk.probability import FreqDist
from nltk.compat import python_2_unicode_compatible
import numpy
from collections import defaultdict
import sys
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score
class ConfusionMatrix(object):

    def __init__(self, reference, test, sort_by_count=False):

        if len(reference) != len(test):
            raise ValueError('Lists are not in sync.')

        # Get a list of all values.
        if sort_by_count:
        	ref_fdist = FreqDist(reference)
        	test_fdist = FreqDist(test)
        	def key(v): return -(ref_fdist[v]+test_fdist[v])
        	values = sorted(set(reference+test), key=key)     
        else:
        	values = sorted(set(reference+test))        	

        # Construct a value->index dictionary
        indices = dict((val,i) for (i,val) in enumerate(values))
        #print(indices)

        # Make a confusion matrix table.
        confusion = [[0 for val in values] for val in values]

        max_conf = 0 # Maximum confusion
        for w,g in zip(reference, test):
            confusion[indices[w]][indices[g]] += 1
            max_conf = max(max_conf, confusion[indices[w]][indices[g]])
        #: A list of all values in ``reference`` or ``test``.
        self._values = values
        #: A dictionary mapping values in ``self._values`` to their indices.
        self._indices = indices
        #: The confusion matrix itself (as a list of lists of counts).
        self._confusion = confusion
        #: The greatest count in ``self._confusion`` (used for printing).
        self._max_conf = max_conf
        #: The total number of values in the confusion matrix.
        self._total = len(reference)
        #: The number of correct (on-diagonal) values in the matrix.
        self._correct = sum(confusion[i][i] for i in range(len(values)))

        #initialize the performance measures to zero
        self._TP = defaultdict(int)
        self._FP = defaultdict(int)
        self._FN = defaultdict(int)
        self._PRE = defaultdict(float)
        self._REC = defaultdict(float)
        self._F1 = defaultdict(float)

        self._PREmic = 0
        self._RECmic = 0
        self._F1mic = 0

        self._PREmac = 0
        self._RECmac = 0
        self._F1mac = 0

        # calculate the performance measures
        self.getTP()
        self.getFP()
        self.getFN()
        self.getPRE()
        self.getREC()
        self.getF1()
        self.getPREmicro()
        self.getRECmicro()
        self.getF1micro()

        self.getPREmacro()
        self.getRECmacro()
        self.getF1macro()
        self.accuracy_score = accuracy_score(reference, test) 

    def printConf(self):
        print(self.pretty_format())

    def pretty_format(self, show_percents=False, values_in_chart=True,
           truncate=None, sort_by_count=False):
        """
        :return: A multi-line string representation of this confusion matrix.
        :type truncate: int
        :param truncate: If specified, then only show the specified
            number of values.  Any sorting (e.g., sort_by_count)
            will be performed before truncation.
        :param sort_by_count: If true, then sort by the count of each
            label in the reference data.  I.e., labels that occur more
            frequently in the reference label will be towards the left
            edge of the matrix, and labels that occur less frequently
            will be towards the right edge.

        @todo: add marginals?
        """
        confusion = self._confusion

        values = self._values
        if sort_by_count:
            values = sorted(values, key=lambda v:
                            -sum(self._confusion[self._indices[v]]))

        if truncate:
            values = values[:truncate]

        if values_in_chart:
            value_strings = ["%s" % val for val in values]
        else:
            value_strings = [str(n+1) for n in range(len(values))]

        # Construct a format string for row values
        valuelen = max(len(val) for val in value_strings)
        value_format = '%' + repr(valuelen) + 's | '
        # Construct a format string for matrix entries
        if show_percents:
            entrylen = 6
            entry_format = '%5.1f%%'
            zerostr = '     .'
        else:
            entrylen = len(repr(self._max_conf))
            entry_format = '%' + repr(entrylen) + 'd'
            zerostr = ' '*(entrylen-1) + '.'

        # Write the column values.
        s = ''
        for i in range(valuelen):
            s += (' '*valuelen)+' |'
            for val in value_strings:
                if i >= valuelen-len(val):
                    s += val[i-valuelen+len(val)].rjust(entrylen+1)
                else:
                    s += ' '*(entrylen+1)
            s += ' |\n'

        # Write a dividing line
        s += '%s-+-%s+\n' % ('-'*valuelen, '-'*((entrylen+1)*len(values)))

        # Write the entries.
        for val, li in zip(value_strings, values):
            i = self._indices[li]
            s += value_format % val
            for lj in values:
                j = self._indices[lj]
                if confusion[i][j] == 0:
                    s += zerostr
                elif show_percents:
                    s += entry_format % (100.0*confusion[i][j]/self._total)
                else:
                    s += entry_format % confusion[i][j]
                if i == j:
                    prevspace = s.rfind(' ')
                    s = s[:prevspace] + '<' + s[prevspace+1:] + '>'
                else: s += ' '
            s += '|\n'

        # Write a dividing line
        s += '%s-+-%s+\n' % ('-'*valuelen, '-'*((entrylen+1)*len(values)))

        # Write a key
        s += '(row = reference; col = test)\n'
        if not values_in_chart:
            s += 'Value key:\n'
            for i, value in enumerate(values):
                s += '%6d: %s\n' % (i+1, value)

        return s

    def getTP(self):
    	for i in range(len(self._values)):
    		self._TP[i] = self._confusion[i][i]

    
    def getFP(self):
    	for j in range(len(self._values)):
    		val = 0
    		for i in range(len(self._values)):
    			if(i != j):
    				val += self._confusion[i][j]	
    		self._FP[j] = val

    def getFN(self):
    	for i in range(len(self._values)):
    		val = 0
    		for j in range(len(self._values)):
    			if(i != j):
    				val += self._confusion[i][j]
    		self._FN[i] = val

    def getPRE(self):
    	for i in range(len(self._values)):
            try:
                self._PRE[i] = float(self._TP[i])/(self._TP[i] + self._FP[i])
            except:
                self._PRE[i] = 1

    def getREC(self):
        for i in range(len(self._values)):
            try:
                self._REC[i] = float(self._TP[i])/(self._TP[i] + self._FN[i])
            except:
                self._REC[i] = 1

    def getF1(self):
    	for i in range(len(self._values)):
            try:
                self._F1[i] = 2.0 * self._PRE[i] * self._REC[i] / (self._PRE[i] + self._REC[i])
            except:
                self._F1[i] = 1


    def getPREmicro(self):    
        try:    
    	   self._PREmic = float(sum(self._TP.itervalues())) / (sum(self._TP.itervalues()) + sum(self._FP.itervalues()))
        except:
            print("Some error occurred!!")
    
    def getRECmicro(self):
    	try:
            self._RECmic = float(sum(self._TP.itervalues())) / (sum(self._TP.itervalues()) + sum(self._FN.itervalues()))
        except:
            print("Some error occurred!!")
    def getF1micro(self):
    	try:
            self._F1mic = 2.0 * self._PREmic * self._RECmic / (self._PREmic + self._RECmic)
        except:
            print("Some error occurred!!")
    def getPREmacro(self):
    	try:
            self._PREmac = float(sum(self._PRE.itervalues())) / len(self._values)
        except:
            print("Some error occurred!!")
    def getRECmacro(self):
    	try:
            self._RECmac = float(sum(self._REC.itervalues())) / len(self._values)
        except:
            print("Some error occurred!!")
    def getF1macro(self):
    	try:
            self._F1mac = 2.0 * self._PREmac * self._RECmac / (self._PREmac + self._RECmac)
        except:
            print("Some error occurred!!")

    #prints the performace measures
    def printResults(self):
        print("Accuracy of the classfier :" + str(self.accuracy_score ))
        print('\n\n')
        print("Precision Recall and F1 measure for respective tags are as follows:")
        for tag in self._values:
            print("\nFor tag = '" + str(tag) +"':")
            print("\tNumber of True Positives : " + str(self._TP[self._indices[tag]]))
            print("\tNumber of False Positives : " + str(self._FP[self._indices[tag]]))
            print("\tNumber of False Negatives : " + str(self._FN[self._indices[tag]]))
            print("\tPrecision : " + str(self._PRE[self._indices[tag]]))
            print("\tRecall : " + str(self._REC[self._indices[tag]]))
            print("\tF1-measure : " + str(self._F1[self._indices[tag]]))


        print('\n\n')
        print("Micro Averaged performance measures for the classfier:")
        print("\t Precision : " + str(self._PREmic))
        print("\t Recall : " + str(self._RECmic))
        print("\t F1-measure : " + str(self._F1mic))
        print('\n\n')
        print("Macro Averaged performance measures:")        
        print("\t Precision : " + str(self._PREmac))
        print("\t Recall  : " + str(self._RECmac))
        print("\t F1-measure  : " + str(self._F1mac))
        print('\n\n')

# given the reference and the test file performs evaluation
def evaluate(reference,test):
    #print('Confusion matrix:')
    conf = ConfusionMatrix(reference, test) 
    conf.printConf()
    conf.printResults()

# returns the list of tags from the given filename
def getTagList(filename):
    myTagList = []
    linenum = 0
    linemetagcnt = defaultdict(int)

    for line in open(filename,'r'):
        linenum += 1
        #print(str(linenum)+" -> "+line)
        line = line.strip()
        if(line):
            mytemplist = []
            for l in line.split(' '):
                l = l.strip()
                if(l):
                    word,tag = '',''
                    try:
                        word, tag = l.split('/')
                    except ValueError:
                        split = l.split('/')
                        word, tag = "/".join(numpy.array(split)[[0,-2]]), split[len(split)-1]
                    mytemplist.append(tag)
            linemetagcnt[linenum] = len(mytemplist)
            myTagList.extend(mytemplist)
    return myTagList, linemetagcnt

if __name__ == '__main__':
    if(len(sys.argv)!=3):
        sys.stderr.write("\nUsage: python eval.py standard_file_name test_file_name\n\n")
    else:        
        tagged_file = sys.argv[1]
        test_file = sys.argv[2]
        #tagged_file = "../Data/Brown_tagged_train.txt"
        #test_file = "test_out_morpho_backoff"
        ref,refnum = getTagList(tagged_file)
        test,testnum = getTagList(test_file)
        print("The performance measures for the test file " + str(test_file) + " are as follows:\n")
        evaluate(ref,test)