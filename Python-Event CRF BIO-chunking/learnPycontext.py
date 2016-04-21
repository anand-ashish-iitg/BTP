#!/usr/bin/env python
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
"""markFindings.py is a program that illustrates the basic use of pyConTextNLP.

At this state, the program assumes that the reports are stored in a SQLite3 database.
By default the database has a table named 'reports' with a field named 'impression'.
These values can be be specified through the command line options."""
from nltk import sent_tokenize, word_tokenize

import sys
import os
from optparse import OptionParser
import sqlite3 as sqlite
import pyConTextNLP
import pyConTextNLP.pyConTextGraph as pyConText
import pyConTextNLP.itemData as itemData
import pyConTextNLP.helpers as helpers
import getpass
import time
import xml.dom.minidom as minidom
import codecs
import networkx as nx
import csv

def modifies(g,n,modifiers):
    pred = g.predecessors(n)
    if( not pred ):
        return False
    pcats = [n.getCategory().lower() for n in pred]
    return bool(set(pcats).intersection([m.lower() for m in modifiers]))

class criticalFinder(object):
    """This is the class definition that will contain the majority of processing
    algorithms for criticalFinder.

    The constructor takes as an argument the name of an SQLite database containing
    the relevant information.
    """
    def __init__(self, options):
        """create an instance of a criticalFinder object associated with the SQLite
        database.
        dbname: name of SQLite database
        """


        # Define queries to select data from the SQLite database
        # this gets the reports we will process
        #self.query1 = '''SELECT %s,%s FROM %s'''%(options.id,options.report_text,options.table)

        #self.conn = sqlite.connect(options.dbname)
        #self.cursor = self.conn.cursor()
        #self.cursor.execute(self.query1)
        #self.reports = self.cursor.fetchall()        
        self.reports = []

        for root, dirs, filenames in os.walk("developmentData"):
            for f in filenames:
                fil = open("developmentData/"+f,"r")
                self.reports.append(fil.read())

        print "number of reports to process",len(self.reports)
        #print self.reports[0]
        self.document = pyConText.ConTextDocument()

        # mods = itemData.instantiateFromCSVtoitemData("lexical_kb.csv")
        # trgs = itemData.instantiateFromCSVtoitemData("domain_kb.csv")

        probableNegations = itemData.itemData([["can rule out","PROBABLE_NEGATED_EXISTENCE","","forward"], ["cannot be excluded","PROBABLE_NEGATED_EXISTENCE",r"""cannot\sbe\s((entirely|completely)\s)?(excluded|ruled out)""","backward"]])

        critItems = itemData.itemData([['pulmonary embolism','PULMONARY_EMBOLISM',r'''pulmonary\s(artery )?(embol[a-z]+)''',''],['pe','PULMONARY_EMBOLISM',r'''\bpe\b''',''],['embolism','PULMONARY_EMBOLISM',r'''\b(emboli|embolism|embolus)\b''',''],['aortic dissection','AORTIC_DISSECTION','','']])


        # self.modifiers = itemData.itemData()
        # for key in mods.keys():
        #     self.modifiers.prepend(mods[key])

        # self.targets = itemData.itemData()
        # for key in trgs.keys():
        #     self.targets.prepend(trgs[key])

        lex = []
        dom = []
        with open('lexical_kb.csv', 'rb') as f:
            reader = csv.reader(f)
            lst = list(reader)
            for item in lst:
                print item
                lex.append([item[0],item[1],item[2],item[3]])
        with open('domain_kb.csv', 'rb') as f:
            reader = csv.reader(f)
            lst = list(reader)
            for item in lst:
                print item
                dom.append([item[0],item[1],item[2],item[3]])
        print lex
        print 
        print
        print dom

        self.modifiers = itemData.itemData(lex)
        self.targets = itemData.itemData(dom)
        self.allow_uncertainty = True

    def initializeOutput(self,lfile,dfile):
        """Provides run specific information for XML output file"""
        self.outString  =u"""<?xml version="1.0"?>\n"""
        self.outString +=u"""<markup>\n"""
        self.outString +=u"""<pyConTextNLPVersion> %s </pyConTextNLPVersion>\n"""%pyConTextNLP.__version__
        self.outString +=u"""<operator> %s </operator>\n"""%getpass.getuser()
        self.outString +=u"""<date> %s </date>\n"""%time.ctime()
        self.outString +=u"""<lexicalFile> %s </lexicalFile>\n"""%lfile
        self.outString +=u"""<domainFile> %s </domainFile>\n"""%dfile

    def closeOutput(self):
        self.outString +=u"""</markup>\n"""
    def getOutput(self):
        return self.outString
    def analyzeReport(self, report ):
        """
        given an individual radiology report, creates a pyConTextGraph
        object that contains the context markup
        report: a text string containing the radiology reports
        """
        print report
        context = self.document
        targets = self.targets
        modifiers = self.modifiers
        sentences = sent_tokenize(report)
        count = 0
        for s in sentences:
            #print s
            markup = pyConText.ConTextMarkup()
            markup.setRawText(s)
            markup.cleanText()
            markup.markItems(modifiers, mode="modifier")
            markup.markItems(targets, mode="target")
            markup.pruneMarks()
            markup.dropMarks('Exclusion')
            markup.applyModifiers()
            #context.pruneModifierRelationships()
            #context.dropInactiveModifiers()
            print markup
            self.outString += context.getXML()+u"\n"
            context.addMarkup(markup)
            count += 1
        context.computeDocumentGraph()
        #print "*"*42
        #print context.getXML(currentGraph=False)
        #print "*"*42

    def processReports(self):
        """For the selected reports (training or testing) in the database,
        process each report with peFinder
        """
        count = 0
        for r in self.reports[0:20]:
                self.document = pyConText.ConTextDocument()
                self.currentText = r
                                
                self.analyzeReport(self.currentText)
                self.outString +=u"</case>\n"
                rslts = self.classifyDocumentTargets()
                print rslts
                raw_input('continue')
        print "_"*48
    def classifyDocumentTargets(self):
        rslts = {}
        g = self.document.getDocumentGraph()
        #print 1/0
        targets = [n[0] for n in g.nodes(data = True) if n[1].get("category","") == 'target']

        if( not targets ):
            print "no targets"
            return rslts
        if(self.allow_uncertainty):
            pos_filters = ["definite_existence","probable_existence"]
        else:
            pos_filters = ["definite_existence"]
        for t in targets:
            mods = g.predecessors(t)
            rslts[t] = {}
            if( not mods ): # an unmodified target is disease positive,certain, and acute

                rslts[t]['disease'] = 'Pos'
                rslts[t]['uncertainty'] = 'No'
                rslts[t]['temporality'] = 'New'
            else:
                if (modifies(g,t,pos_filters) and
                    not modifies(g,t,[#"definite_negated_existence",
                                      #"probable_negated_existence",
                                      "future","indication","pseudoneg"])):
                    rslts[t]['disease'] = 'Pos'
                else:
                    rslts[t]['disease'] = 'Neg'
                if( modifies(g,t,["probable_existence",
                                  "probable_negated_existence"]) ):
                    rslts[t]['uncertainty'] = 'Yes'
                else:
                    rslts[t]['uncertainty'] = 'No'
                if( modifies(g,t,["historical"]) ):
                    rslts[t]['temporality'] = 'Old'
                else:
                    if( rslts[t]['disease'] == 'Neg'):
                        rslts[t]['temporality'] = 'NA'
                    else:
                        rslts[t]['temporality'] = 'New'

        return rslts

def getParser():
    """Generates command line parser for specifying database and other parameters"""

    parser = OptionParser()
    # parser.add_option("-b","--db",dest='dbname',
    #                   help='name of SQLite database containing reports to parse')
    parser.add_option("-o","--output",dest="ofile",
                      help="name of file for xml output",default="output")
    parser.add_option("-l","--lexical_kb",dest='lexical_kb',
                      help='name of csv file containing lexical knowledge', default="lexical_kb.csv")
    parser.add_option("-d","--domain_kb",dest='domain_kb',
                      help='name of csv file containing domain knowledge', default="domain_kb.csv")
    # parser.add_option("-t","--table",dest='table',default='reports',
    #                   help='table in database to select data from')
    # parser.add_option("-i","--id",dest='id',default='rowid',
    #                   help='column in table to select identifier from')
    # parser.add_option("-r","--report",dest='report_text',default='impression',
    #                   help='column in table to select report text from')
    parser.add_option("-u","--uncertainty_allowed",dest="allow_uncertainty",
                      action="store_true",default=False)

    return parser

def main():

    parser = getParser()
    (options, args) = parser.parse_args()
    pec = criticalFinder(options)
    pec.initializeOutput(options.lexical_kb, options.domain_kb)
    pec.processReports()
    pec.closeOutput()
    txt = pec.getOutput()
#    f0 = codecs.open(options.ofile+".xml",encoding=pec.document.getUnicodeEncoding(),mode="w")
#    f0.write(txt)
#    f0.close()

#    try:
#        xml = minidom.parseString(txt)
#        print xml.toprettyxml(encoding=pec.document.getUnicodeEncoding())
#        f0 = codecs.open(options.ofile+"_pretty.xml",encoding=pec.document.getUnicodeEncoding(),mode="w")
#        f0.write(xml.toprettyxml(encoding=pec.document.getUnicodeEncoding()))
#        f0.close()
#    except Exception, error:
#        print "could not prettify xml"
#        print error


if __name__=='__main__':

    main()

