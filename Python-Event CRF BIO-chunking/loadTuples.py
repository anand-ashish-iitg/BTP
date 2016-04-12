import pickle
import os  

def load(trainOrTest):
	objs = []
	for root, dirs, filenames in os.walk("dumpPickles/" + trainOrTest):
		for f in filenames:
			#print root,dirs,f
			fpkl=open(root+"/"+f, 'rb')
			objs.extend(pickle.load(fpkl))
			fpkl.close()
	return objs    

