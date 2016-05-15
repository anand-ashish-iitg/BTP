import nltk
import os.path
import os  
from nltk import sent_tokenize, word_tokenize
from parseGoldTime import getSpans
from parseMedTime import getMedSpans
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

        medFilePath =  "MedTime-output/"+trainOrTest+"/" + fileName + ".ann"
        medFileExists = False
        if(os.path.exists(medFilePath)):
            medFileExists = True

        if(fileExists and medFileExists):
            spans = getSpans(filePath)     
            medspans = getMedSpans(fileName,trainOrTest)


           
            # get the text from file
            with  open("raw_data/" + trainOrTest + "/" + fileName,"r") as fil:
                text =  fil.read()
                
                # for cords in spans:
                #     print "File : " +  fileName + " class: " + cords[2] + " expression : " + text[cords[0]:cords[1]]                
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
                            medlabel = "O"   
                            medclass = "DATE"
                            Class = "DATE"                         
                            for cord in spans:
                                #print cord[0],cord[1]                                    
                                if(word_offset==cord[0]):
                                    #print "B-EENT word = " + word + " spanned string = " + text[cord[0]:cord[1]] +  " cord[0],cord[1] = " + str(cord[0])+","+str(cord[1]) + "word_offset,run off  = " +str(word_offset) +"," + str(running_offset)
                                    Class =  cord[2]
                                    label = "B-TIMEX"
                                    break
                                elif(word_offset>cord[0] and (running_offset)<=cord[1]):
                                    #print " I-EVENT word = " + word + " spanned string = " + text[cord[0]:cord[1]] +  " cord[0],cord[1] = " + str(cord[0])+","+str(cord[1]) + "word_offset,run off  = " +str(word_offset) +"," + str(running_offset)
                                    Class =  cord[2]
                                    label = "I-TIMEX" 
                                    break    

                            for cord in medspans:
                                #print cord[0],cord[1]                                    
                                if(word_offset==cord[0]):
                                    #print "B-EENT word = " + word + " spanned string = " + text[cord[0]:cord[1]] +  " cord[0],cord[1] = " + str(cord[0])+","+str(cord[1]) + "word_offset,run off  = " +str(word_offset) +"," + str(running_offset)
                                    medclass =  cord[2]
                                    medlabel = "B-TIMEX"
                                    break
                                elif(word_offset>cord[0] and (running_offset)<=cord[1]):
                                    #print " I-EVENT word = " + word + " spanned string = " + text[cord[0]:cord[1]] +  " cord[0],cord[1] = " + str(cord[0])+","+str(cord[1]) + "word_offset,run off  = " +str(word_offset) +"," + str(running_offset)
                                    medclass =  cord[2]
                                    medlabel = "I-TIMEX" 
                                    break              
                        except ValueError:
                            label = "O"
                            medlabel = "O"
                            print "Exception with word = " + word
                            continue
                        
                        #print word,label
                        offsets.append((word, pos[1], label, word_offset, running_offset-1, fileName, medlabel, Class, medclass))
                        #print text[word_offset:running_offset] + " " + word
                    textOffsets.append(offsets)
                f=open("TimeSpandumpPickles/" + trainOrTest + "/" + fileName + "-" + pickleFile, 'wb')
                pickle.dump(textOffsets, f)
                f.close()
                return textOffsets
        else:
            print "File does not exist!!"
    except Exception as e:
        # print "There was some problem when processing file: " +fileName
        print e
