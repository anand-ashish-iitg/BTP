from tokenizer import getTuples
import os  

istrain = False
filenum = 0
if(istrain):
	for root, dirs, filenames in os.walk("raw_data/train"):
		for f in filenames:
			filenum += 1
			print "Processing : " + f + " filenum = "  + str(filenum)
			getTuples(f,"train_tuples.pkl","train")

else:
	for root, dirs, filenames in os.walk("raw_data/test"):
		for f in filenames:
			filenum += 1
			print "Processing : " + f + " filenum = "  + str(filenum)
			getTuples(f,"test_tuples.pkl","test")   
#print "Processing" +fileName1
#getTuples(fileName3,'tuples.pkl')
'''
print "Processing" + fileName3
getTuples(fileName3,'test_tuples.pkl',"test")'''
