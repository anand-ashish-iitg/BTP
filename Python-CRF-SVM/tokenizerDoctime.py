import nltk
import os.path
import os  
from nltk import sent_tokenize, word_tokenize
from parseGoldDocTime import getSpans
import pickle
#process annotatation given to have the labels for words based on the span



def getTuples(fileName, pickleFile, trainOrTest = "train", _len=len):
    try:
        # get the filename that are present        
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
            normalizedDict = {}
            cuiDict = {}
            tuiDict = {}

            # process the cTakes processed date to form dictionary to get the normalized value
            with open("ctakesProcessed/" + trainOrTest+"/" + fileName + ".xml-normalized","r") as fil:
                for line in fil:
                    tok = line.split(',')
                    if(len(tok)==2):
                        normalizedDict[tok[0]] = tok[1]


            # process the cTakes processed date to form dictionary to get the CUI
            with open("ctakesProcessed/" + trainOrTest+"/" + fileName + ".xml-cui","r") as fil:
                for line in fil:
                    tok = line.split(',')
                    if(len(tok)==2):
                        cuiDict[tok[0]] = tok[1]


            with open("ctakesProcessed/" + trainOrTest+"/" + fileName + ".xml-tui","r") as fil:
                for line in fil:
                    tok = line.split(',')
                    if(len(tok)==2):
                        tuiDict[tok[0]] = tok[1]        

            '''print normalizedDict
            print cuiDict
            print tuiDict    '''

            # get the text from file
            with  open("raw_data/" + trainOrTest + "/" + fileName,"r") as fil:
                text =  fil.read()
                
                # get the list of all the sentences
                sents = sent_tokenize(text)
                index = text.index
                running_offset = 0
                textOffsets = []

                #iterate over all the sentences
                for line in sents:
                    print "."
                    #print line
                    #get tokens in a sentence
                    words = word_tokenize(line)
                    words = ['"' if (x == "''" or x =="``")  else x for x in words ]
                    #print words 

                    # get the POS tags of the words
                    posTaggedTokens = nltk.pos_tag(words)
                 
                    offsets = []        
                    ind = -1

                    for word in words:
                        #print word
                        #print _len(word)


                        try:
                            ind += 1
                            # find the position of the word
                            word_offset = index(word, running_offset)
                            pos = posTaggedTokens[ind]
                            word_len = _len(word)
                            running_offset = word_offset + word_len
                            label = "O"
                            Type = "N/A"
                            Degree = "N/A"
                            Polarity = "POS"
                            Modality = "ACTUAL"
                            Aspect = "N/A"
                            Doctime = "OVERLAP"
                            for cord in spans:
                                #print cord[0],cord[1]                                    
                                if(word_offset==cord[0]):
                                    #print "B-EVENT word = " + word + " spanned string = " + text[cord[0]:cord[1]] +  " cord[0],cord[1] = " + str(cord[0])+","+str(cord[1]) + "word_offset,run off  = " +str(word_offset) +"," + str(running_offset)
                                    label = "B-EVENT"
                                    Type =  cord[2]
                                    Degree = cord[3]
                                    Polarity = cord[4]
                                    Modality = cord[5]
                                    Aspect = cord[6]
                                    Doctime = cord[7]
                                    break
                                elif(word_offset>cord[0] and (running_offset)<=cord[1]):
                                    #print " I-EVENT word = " + word + " spanned string = " + text[cord[0]:cord[1]] +  " cord[0],cord[1] = " + str(cord[0])+","+str(cord[1]) + "word_offset,run off  = " +str(word_offset) +"," + str(running_offset)
                                    label = "I-EVENT"
                                    Type =  cord[2]
                                    Degree = cord[3]
                                    Polarity = cord[4]
                                    Modality = cord[5]
                                    Aspect = cord[6]
                                    Doctime = cord[7]
                                    break                
                        except ValueError:
                            label = "O"
                            print "Exception with word = " + word
                            continue
                        norm = word
                        if(normalizedDict.has_key(word)):
                            norm = normalizedDict[word]

                        cui = 'none'
                        if(cuiDict.has_key(word)):
                            cui = cuiDict[word]

                        tui = 'none'
                        if(tuiDict.has_key(word)):
                            tui = tuiDict[word]

                        #print word,label
                        offsets.append((word, pos[1], norm, cui, tui, label, word_offset, running_offset-1, fileName, Type, Degree, Polarity, Modality, Aspect, Doctime))
                        #print text[word_offset:running_offset] + " " + word
                    textOffsets.append(offsets)
                f=open("DocTimedumpPickles/" + trainOrTest + "/" + fileName + "-" + pickleFile, 'wb')
                pickle.dump(textOffsets, f)
                f.close()
                #return textOffsets
        else:
            print "File does not exist!!"
    except:
        print "There was some problem when processing file: " +fileName


# getTuples("ID001_clinic_001", "train_tuples.pkl", "train")

#print spans