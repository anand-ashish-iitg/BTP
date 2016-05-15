from xml.dom.minidom import parse
import xml.dom.minidom
import re

def getSpans(filename):
   """parses the anafora files to get the properties for entities from gold annotated data

    Args:
        filename: the file to be parsed
    Returns:
        the spans and properties of all the tokens 
    """
   # Open XML document using minidom parser
   try:
      DOMTree = xml.dom.minidom.parse(filename)
      data = DOMTree.documentElement

      annotations = data.getElementsByTagName("annotations")[0]

      entities = annotations.getElementsByTagName("entity")
      spans = []
      # get properties of each entity.
      for entity in entities:
         type = entity.getElementsByTagName('type')[0]
         #print "Type: %s" % type.childNodes[0].data   
         if(type.childNodes[0].data == "EVENT"):
            span = entity.getElementsByTagName('span')[0]
            cords = re.split(';|,',span.childNodes[0].data)
            #print cords
            properties = entity.getElementsByTagName('properties')[0]
            TypeNode = properties.getElementsByTagName('Type')[0]
            Type = TypeNode.childNodes[0].data
            
            DegreeNode = properties.getElementsByTagName('Degree')[0]
            Degree = DegreeNode.childNodes[0].data

            PolarityNode = properties.getElementsByTagName('Polarity')[0]
            Polarity = PolarityNode.childNodes[0].data

            ContextualModalityNode = properties.getElementsByTagName('ContextualModality')[0]
            ContextualModality = ContextualModalityNode.childNodes[0].data

            ContextualAspectNode = properties.getElementsByTagName('ContextualAspect')[0]
            ContextualAspect = ContextualAspectNode.childNodes[0].data
            spans.append((int(cords[0]),int(cords[1]),Type,Degree,Polarity,ContextualModality,ContextualAspect))
            
      return spans      
   except:
      print "Error parsing file : " +  filename
      raise

#print getSpans("/home/subh/Desktop/BTP/Python/gold_annotated/train/ID001_clinic_001/ID001_clinic_001.Temporal-Relation.gold.completed.xml")      