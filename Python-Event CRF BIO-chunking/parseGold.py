from xml.dom.minidom import parse
import xml.dom.minidom
import re

def getSpans(filename):
   # Open XML document using minidom parser
   try:
      DOMTree = xml.dom.minidom.parse(filename)
      data = DOMTree.documentElement

      annotations = data.getElementsByTagName("annotations")[0]

      entities = annotations.getElementsByTagName("entity")
      spans = []
      # Print detail of each movie.
      for entity in entities:
         type = entity.getElementsByTagName('type')[0]
         #print "Type: %s" % type.childNodes[0].data   
         if(type.childNodes[0].data == "EVENT"):
            span = entity.getElementsByTagName('span')[0]
            cords = re.split(';|,',span.childNodes[0].data)
            #print cords
            spans.append((int(cords[0]),int(cords[1])))

      return spans      
   except:
      print "Error parsing file : " +  filename
      raise

print getSpans("/home/subh/Desktop/BTP/Python/gold_annotated/train/ID001_clinic_001/ID001_clinic_001.Temporal-Relation.gold.completed.xml")