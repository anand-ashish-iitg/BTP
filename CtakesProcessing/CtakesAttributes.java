import java.util.*;
import java.io.File;
import java.io.PrintWriter;

import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.DocumentBuilder;

import org.w3c.dom.Document;
import org.w3c.dom.NodeList;
import org.w3c.dom.Node;
import org.w3c.dom.Element;
public class CtakesAttributes {
	
  public static void doPreprocessing(String inputFileName,boolean istrain){
    /*does the preprocessing of the cTAKES generated output to extract the cui, tui and normalized values.

      Args:
        inputFileName: name of the file to be processes
        istrain: whether its training or testing data
    */
    try {
       //String inputFileName = "Ctakesoutput/1.xml";
       Map<String, String> cuidMap = new HashMap<String, String>();
       Map<String, String> tuidMap = new HashMap<String, String>();
       Map<String, String> normalizedMap = new HashMap<String, String>();
       Map<String, String> fsArrayMap = new HashMap<String, String>();
       Map<String, String> umlsCuiMap = new HashMap<String, String>();
       Map<String, String> umlsTuiMap = new HashMap<String, String>();
       File inputFile;
       if(istrain)
    	   inputFile = new File("Ctakesoutput/train/" + inputFileName);
       else
    	   inputFile = new File("Ctakesoutput/test/" + inputFileName);
       
       DocumentBuilderFactory dbFactory 
          = DocumentBuilderFactory.newInstance();
       DocumentBuilder dBuilder = dbFactory.newDocumentBuilder();
       Document doc = dBuilder.parse(inputFile);
       doc.getDocumentElement().normalize();

       
       NodeList fileList = doc.getElementsByTagName("uima.cas.Sofa");
       Node fileNode = fileList.item(0);
       Element fElement = (Element) fileNode;
       String fileContent = fElement.getAttribute("sofaString");
       //System.out.print(fileContent);
       NodeList ret = doc.getElementsByTagName("org.apache.ctakes.typesystem.type.syntax.WordToken");
       NodeList[] nodeList = new NodeList[7];
       // get all instances of concept mention
       nodeList[0] = doc.getElementsByTagName("org.apache.ctakes.typesystem.type.textsem.AnatomicalSiteMention");
       nodeList[1] = doc.getElementsByTagName("org.apache.ctakes.typesystem.type.textsem.DiseaseDisorderMention");
       nodeList[2] = doc.getElementsByTagName("org.apache.ctakes.typesystem.type.textsem.LabMention");
       nodeList[3] = doc.getElementsByTagName("org.apache.ctakes.typesystem.type.textsem.MedicationEventMention");
       nodeList[4] = doc.getElementsByTagName("org.apache.ctakes.typesystem.type.textsem.MedicationMention");
       nodeList[5] = doc.getElementsByTagName("org.apache.ctakes.typesystem.type.textsem.ProcedureMention");
       nodeList[6] = doc.getElementsByTagName("org.apache.ctakes.typesystem.type.textsem.SignSymptomMention");
       
       NodeList fsArray = doc.getElementsByTagName("uima.cas.FSArray");
       NodeList umlsConcept = doc.getElementsByTagName("org.apache.ctakes.typesystem.type.refsem.UmlsConcept");
       for(int  i = 0;i<fsArray.getLength();i++ ){
    	   Node nNode = fsArray.item(i);
  		   if (nNode.getNodeType() == Node.ELEMENT_NODE) {
  	             Element eElement = (Element) nNode;
  	             if(nNode.hasChildNodes()){
  	            	 String umlsid = eElement.getElementsByTagName("i").item(0).getTextContent(); 
  	            	 //System.out.println("Upmls id u = " + umlsid);
  		             fsArrayMap.put(eElement.getAttribute("_id"),umlsid);
  		             //System.out.println("fsarrayid is = " +eElement.getAttribute("_id"));
  		             //System.out.println("pehle fsarray  = "+ fsArrayMap.get(eElement.getAttribute("_id")));
  	             }	            
  		   }

       }
       
       for(int  i = 0;i<umlsConcept.getLength();i++ ){
    	   Node nNode = umlsConcept.item(i);
  		   if (nNode.getNodeType() == Node.ELEMENT_NODE) {
  	             Element eElement = (Element) nNode;
  	            	 String cui = eElement.getAttribute("cui");
  	            	 String tui = eElement.getAttribute("tui");
  	            	 //System.out.println("cui =  " + cui + " tui = " + tui + " id = " + eElement.getAttribute("_id"));
  		             umlsCuiMap.put(eElement.getAttribute("_id"),cui);
  		             umlsTuiMap.put(eElement.getAttribute("_id"),tui);
  	             	 //System.out.println(umlsCuiMap.get(eElement.getAttribute("_id")));

  		   }

       }
       
       for(int len=0; len<nodeList.length;len++){
    	   int listlen = nodeList[len].getLength();
    	   for(int temp = 0; temp<listlen;temp++){
    		   Node nNode = nodeList[len].item(temp);
    		   if (nNode.getNodeType() == Node.ELEMENT_NODE) {
    	             Element eElement = (Element) nNode;
    	             if(eElement.hasAttribute("_ref_ontologyConceptArr")){
    	            	 String fsArrayid = eElement.getAttribute("_ref_ontologyConceptArr");
    	            	 //System.out.println("fsArrayid = " + fsArrayid);
    	            	 String umlsid = fsArrayMap.get(fsArrayid);
    	            	 //System.out.println("umlsid = " +  umlsid);
    	            	 String cui = umlsCuiMap.get(umlsid);
    	            	 String tui = umlsTuiMap.get(umlsid);
    	            	 String tokenText = fileContent.substring(Integer.parseInt(eElement.getAttribute("begin")), Integer.parseInt(eElement.getAttribute("end")));
    	             	 cuidMap.put(tokenText, cui);
    	             	 tuidMap.put(tokenText, tui);

    	             }
    	             
    	       }
    	   }
       }
       
       System.out.println("----------------------------");
       for (int temp = 0; temp < ret.getLength(); temp++) {
          Node nNode = ret.item(temp);
          if (nNode.getNodeType() == Node.ELEMENT_NODE) {
             Element eElement = (Element) nNode;
             String tokenText = fileContent.substring(Integer.parseInt(eElement.getAttribute("begin")), Integer.parseInt(eElement.getAttribute("end"))) ;
             normalizedMap.put(tokenText, eElement.getAttribute("normalizedForm"));                           
          }
       }

       System.out.println(" Map Elements");
       System.out.print("\t" + normalizedMap);

       System.out.println("CUI Map Elements");
       System.out.print("\t" + cuidMap);

       System.out.println("TUI Map Elements");
       System.out.print("\t" + tuidMap);
       
       PrintWriter writer;
       if(istrain){
    	    writer = new PrintWriter("ctakesProcessed/train/" +  inputFileName + "-normalized", "UTF-8");
       }else{
    	   	writer = new PrintWriter("ctakesProcessed/test/" +  inputFileName + "-normalized", "UTF-8");
       }       
       for (Map.Entry<String,String> entry : normalizedMap.entrySet()) {
    	   writer.println(entry.getKey() + "," + entry.getValue());
       }       
       writer.close();
       
       if(istrain){
   	    writer = new PrintWriter("ctakesProcessed/train/" +  inputFileName + "-cui", "UTF-8");
      }else{
   	   	writer = new PrintWriter("ctakesProcessed/test/" +  inputFileName + "-cui", "UTF-8");
      } 
       for (Map.Entry<String,String> entry : cuidMap.entrySet()) {
    	   writer.println(entry.getKey() + "," + entry.getValue());
       }
       writer.close();
       
       if(istrain){
	   	    writer = new PrintWriter("ctakesProcessed/train/" +  inputFileName + "-tui", "UTF-8");
	   }else{
	   	   	writer = new PrintWriter("ctakesProcessed/test/" +  inputFileName + "-tui", "UTF-8");
	   }        
       for (Map.Entry<String,String> entry : tuidMap.entrySet()) {
    	   writer.println(entry.getKey() + "," + entry.getValue());
       }
       writer.close();
    } catch (Exception e) {
       e.printStackTrace();
    }
 }
	
	 public static void main(String... a){
		   
		 boolean istrain = false;
		 File dir;
		 if(istrain){
		 	 dir = new File("Ctakesoutput/train");
		 }else{
			 dir = new File("Ctakesoutput/test");
		 }
		   File[] directoryListing = dir.listFiles();
		   if (directoryListing != null) {
		     for (File child : directoryListing) {
		    	 System.out.println("Processing file :" + child.getName());
		    	 doPreprocessing(child.getName(),istrain);
		     }
		   } else {
		     System.out.println("The mentioned directory is not valid!!. Please try again.");
		   }
	 }
}

