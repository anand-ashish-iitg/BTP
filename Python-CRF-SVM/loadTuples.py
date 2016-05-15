import pickle
import os  

def load(trainOrTest):
	"""loads the date dumped in pickle files for each file into a vector

    Args:
        trainOrTest: whether it is training data or testing data
    Returns:
        the vector containing data in dumped pickle for each file
    """
	objs = []
	for root, dirs, filenames in os.walk("dumpPickles/" + trainOrTest):
		for f in filenames:
			#print root,dirs,f
			fpkl=open(root+"/"+f, 'rb')
			objs.extend(pickle.load(fpkl))
			fpkl.close()
	return objs    

def load2(trainOrTest):
	"""loads the date dumped in pickle files for each file into a vector for event spans

    Args:
        trainOrTest: whether it is training data or testing data
    Returns:
        the vector containing data in dumped pickle for each file
    """
	objs = []
	for root, dirs, filenames in os.walk("dumpPickles/" + trainOrTest):
		for f in filenames:
			#print root,dirs,f
			fpkl=open(root+"/"+f, 'rb')
			objs.extend(pickle.load(fpkl))
			fpkl.close()
	return objs    

def load3(trainOrTest):
	"""loads the date dumped in pickle files for each file into a vector for event attributes

    Args:
        trainOrTest: whether it is training data or testing data
    Returns:
        the vector containing data in dumped pickle for each file
    """
	objs = []
	for root, dirs, filenames in os.walk("SVMdumpPickles/" + trainOrTest):
		for f in filenames:
			#print root,dirs,f
			fpkl=open(root+"/"+f, 'rb')
			objs.extend(pickle.load(fpkl))
			fpkl.close()
	return objs  

	
def load4(trainOrTest):
	"""loads the date dumped in pickle files for each file into a vector for doctime relation

    Args:
        trainOrTest: whether it is training data or testing data
    Returns:
        the vector containing data in dumped pickle for each file
    """
	objs = []
	for root, dirs, filenames in os.walk("DocTimedumpPickles/" + trainOrTest):
		for f in filenames:
			#print root,dirs,f
			fpkl=open(root+"/"+f, 'rb')
			objs.extend(pickle.load(fpkl))
			fpkl.close()
	return objs  

def load5(trainOrTest):
	"""loads the date dumped in pickle files for each file into a vector for time attributes

    Args:
        trainOrTest: whether it is training data or testing data
    Returns:
        the vector containing data in dumped pickle for each file
    """
	objs = []
	for root, dirs, filenames in os.walk("TimedumpPickles/" + trainOrTest):
		for f in filenames:
			#print root,dirs,f
			fpkl=open(root+"/"+f, 'rb')
			objs.extend(pickle.load(fpkl))
			fpkl.close()
	return objs  

def load6(trainOrTest):
	"""loads the date dumped in pickle files for each file into a vector for time span

    Args:
        trainOrTest: whether it is training data or testing data
    Returns:
        the vector containing data in dumped pickle for each file
    """
	objs = []
	for root, dirs, filenames in os.walk("TimeSpandumpPickles/" + trainOrTest):
		for f in filenames:
			#print root,dirs,f
			fpkl=open(root+"/"+f, 'rb')
			objs.extend(pickle.load(fpkl))
			fpkl.close()
	return objs  