import nltk
import os.path
import os  
from nltk import sent_tokenize, word_tokenize
from parseGoldTime import getSpans
import pickle
#process annotatation given to have the labels for words based on the span



def getTuples(fileName, pickleFile, trainOrTest = "train", _len=len):
    try:
        filePath1 ="gold_annotated/" + trainOrTest +"/"+ fileName +"/" + fileName + ".Temporal-Relation.gold.completed.xml"
        filePath2 ="gold_annotated/" + trainOrTest +"/"+ fileName +"/" + fileName + ".Temporal-Entity.gold.completed.xml"
        filePath = ""
        fileExists = False
        if(os.path.exists(filePath1)):
            fileExists = True
            filePath = filePath1
        elif(os.path.exists(filePath2)):
            fileExists = True
            filePath = filePath2

        if(fileExists):
            spans = getSpans(filePath)        
           
            # get the text from file
            with  open("raw_data/" + trainOrTest + "/" + fileName,"r") as fil:
                text =  fil.read()
                
                for cords in spans:
                    print "File : " +  fileName + " class: " + cords[2] + " expression : " + text[cords[0]:cords[1]]

                
                               #return textOffsets
        else:
            print "File does not exist!!"
    except:
        print "There was some problem when processing file: " +fileName
filenum =-1
for root, dirs, filenames in os.walk("raw_data/train"):
        for f in filenames:
            filenum += 1
            print "Processing : " + f + " filenum = "  + str(filenum)
            getTuples(f,"TimePickle.pkl","train")

#print spans