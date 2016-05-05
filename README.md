# BTP #
###Dependencies###
- cTAKES
- MedTime
- nltk
- Sentlex
- CRFSuite
- Scikit learn


##Preprocessing##

- For preprocessing we have used cTAKES, MedTime and few java scripts. 
- We first process the files using cTAKES to get domain specific knowledge required to adapt the temporal evaluation challenge to clinical domain. 
- We process the cTAKES output to extract features like Medlex normalized value, Semantic type(tui), Concept Unique identifier(cui), whether a token was a concept/medication/sign/symptom, etc. 
- cTAKES generated output is then parsed to get the relevant information. 
- We use MedTime to get the corresponding span and type output which is further used in our Hybrid model as a feature.
- We used SentLex to get the polarity of words.

###Steps for preprocessing:###

####Running MedTime####

- Change current directry to MedTime-1.0.2

	$ cd MedTime-1.0.2

- Run the following java command to open Collection processing engine for MedTime.

	$ java -Xms512M -Xmx2000M -cp resources:desc:descsrc:medtaggerdescsrc:MedTime-1.0.2.jar org.apache.uima.tools.cpm.CpmFrame

- After the window open load the CPE descriptor from the File menu.
	File => Open CPE descriptor => Select the File MedTime-1.0.2/desc/medtimedesc/collection_processing_engine/MedTimeCPE.xml
- Once the Aggregate processing engine is loaded. Select the input and the output directory turn by turn first for training files and then for testing files.

- This requires the training and the testing files to be present in the input folder and the outputs are generated in the output folder specified. We specify input folders as "testdata/medtimetest/input/train" and "testdata/medtimetest/input/test". The corresponding output folders are specified as "testdata/medtimetest/output/train" and "testdata/medtimetest/input/test".

- Now, click on Run button on the bottom centre of the window turn by turn for test and the train data.

- The have the TIMEML annotated date with the temporal expressions and their type which is to be used as feature in our Hybrid model for Temporal Span reasoning.


####Using cTAKES####

- Move the the home directory of the project and change current directry to apache-ctakes-3.2.2

	$ cd apache-ctakes-3.2.2

- UMLS dictionary access in cTAKES requires LICENCE. To request a license and creat an account visit "https://uts.nlm.nih.gov/license.html".

- Add the UMLS username and password to bin/runctakesCPE_train.sh and bin/runctakesCPE_test.sh by replacing UMLS_USERNAME and UMLS_PASSWORD present at the end of these files in the java command.

- Copy the training and testing files in folders "input/train" and "input/test" respectively.

- Run for training files:

	$./bin/runctakesCPE_train.sh

- Run for testing files:

	$./bin/runctakesCPE_test.sh

- The output is generated in output/train and output/test

####Parsing cTAKES output####

- Move to the home directory and then change current directory to CtakesProcessing.

	$ cd CtakesProcessing

- Copy the cTAKES generated output in the last step to folders Ctakesoutput/train and Ctakesoutput/test.

	$ cp  -r ../apache-ctakes-3.2.2/output/train Ctakesoutput/train
	$ cp  -r ../apache-ctakes-3.2.2/output/test Ctakesoutput/test

- Compile the file CtakesAttributes.java

	$ javac CtakesAttributes.java

- Execute the program in CtakesAttributes

	$ java CtakesAttributes

- This will generate parsed outputs in ctakesProcessed/train and ctakesProcessed/train.


####Preprocessing the gold annotated and the raw data to get the tags required for ttaining and testing####

- Move to the home directory of the project and change the current directory to Python-CRF-SVM.

	$ cd Python-CRF-SVM

- Create folders and copy the necessary files
	- ctakesProcessed: cTAKES processed output already generated earlier.
	- gold_annotated: the annotated train and test data in corresponding train and test folders.
	- MedTime-output: the output generated from MedTime that we have already obtained.
	- raw_data: the raw text files in subdirectories train and test respectively.

- Create empty directories each with subfolders train and test.
	- MedTimeTemp-output: Used for converting the MedTime output from TimeML format to Anafora format for preprocessing.
	- DocTimedumpPickles: for dump files required for DocTime relation identification.
     	- MedTimeTemp-output: Used for converting the MedTime output from TimeML format to Anafora format for preprocessing.
	- dumpPickles: for preprocessed data for Event Span detection.

	- SVMdumpPickles: Used for preprocessed data for SVM Event attributes detection.
	- System-output: Used to generate the finall processed Anafora format files with the identified events, temporal expressions and 		relations.
	- TimedumpPickles: for preprocessed data used by Time attributes detection.
	- TimeSpandumpPickles: for preprocessed data used by Time Span detection.

- Run the preproceesing files each to get the preprocessed information in the above created folders first with for test files with no argument passed and then with train passed as an argument to process training files.

	For the test file:

	$ python dumpTuples.py
	$ python dumpTuplesSVM.py
	$ python dumpTuplesTime.py
	$ python dumpTuplesDoctime.py
	$ python dumpTuplesTimeSpan.py

	For the training files:

	$ python dumpTuples.py train
	$ python dumpTuplesSVM.py train
	$ python dumpTuplesTime.py train
	$ python dumpTuplesDoctime.py train
	$ python dumpTuplesTimeSpan.py train

- This completes the preprocessing part of our project.


####Training and Testing using CRF and SVM####

#####Timex Span Identification:#####

	$ python trainCRF-TimexSpan.py
	$ python testCRF-TimexSpan.py
	$ python trainSVM-TimexSpan.py
	$ python testSVM-TimexSpan.py

#####Timex Attribute Classification:#####
The Timex Span detection using CRF is used in attribute classification. So, run TimexSpan identification using CRF first and then attribute detection.

######Type:######

	$ python trainSVM-TimexType.py
	$ python testSVM-TimexType.py

#####Event Span Identification:#####

######Train CRF for Event Span detection######

	$ python trainCRF-EventSpan.py
	$ python testCRF-EventSpan.py
	$ python trainSVM-EventSpan.py
	$ python testSVM-EventSpan.py


#####Event Attribute Classification:#####

The Event Span detection using CRF is used in attribute classification. So, run EventSpan identification using CRF first and then attribute detection.

######Type:######

	$ python trainSVM-EventType.py
	$ python testSVM-EventType.py	

######Modality:######

	$ python trainSVM-Modality.py
	$ python testSVM-Modality.py

######Polarity:######

	$ python trainSVM-Polarity.py
	$ python testSVM-Polarity.py

######Degree:######

	$ python trainSVM-Degree.py
	$ python testSVM-Degree.py


#####Document Relation Identification:#####

Event Span identification and Timex span identification to be performed before this step.

	$ python trainCRF-DoctimeRelation.py
	$ python testCRF-DoctimeRelation.py
