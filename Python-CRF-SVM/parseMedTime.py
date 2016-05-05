from xml.dom.minidom import parse
import xml.dom.minidom
import re

def getMedSpans(filename,trainOrtest="test"):
   # Open XML document using minidom parser
   try:
      medtimeRoot = "MedTime-output/"+trainOrtest+"/"
      medtimeTempRoot = "MedTimeTemp-output/"+trainOrtest+"/"
      startStr = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<data>\n\n<schema path=\"./\" protocol=\"file\">temporal-schema.xml</schema>\n\n<annotations>\n"
      endStr = "\n</annotations>\n</data>"
      text = ""  
      with open(medtimeRoot+ filename + ".ann","rb") as fil:
         text = fil.read()
      with open(medtimeTempRoot + filename + ".xml","wb") as fil:
         fil.write(startStr)
         fil.write(text)
         fil.write(endStr)
         


      DOMTree = xml.dom.minidom.parse(medtimeTempRoot+filename +".xml")
      data = DOMTree.documentElement 
      spans = []

      timex = data.getElementsByTagName("TIMEX3")
      for tim in timex:
         start =  tim.getAttribute('start')
         end =  tim.getAttribute('end')
         typed =  tim.getAttribute('type')
         val = tim.getAttribute('val')
         tex = tim.getAttribute('text')
         spans.append((int(start)-1,int(end)-1,typed,val,tex))


      # entities = annotations.getElementsByTagName("entity")
      # Print detail of each movie.
      # for entity in entities:
      #    type = entity.getElementsByTagName('type')[0]
      #    #print "Type: %s" % type.childNodes[0].data   
      #    if(type.childNodes[0].data == "TIMEX3"):
      #       span = entity.getElementsByTagName('span')[0]
      #       cords = re.split(';|,',span.childNodes[0].data)
      #       #print cords
      #       properties = entity.getElementsByTagName('properties')[0]
      #       ClassNode = properties.getElementsByTagName('Class')[0]
      #       Class = ClassNode.childNodes[0].data
            
      #       spans.append((int(cords[0]),int(cords[1]),Class))
            
      return spans      
   except:
      print "Error parsing file : " +  filename
      return []

# print getMedSpans("ID004_clinic_010")      