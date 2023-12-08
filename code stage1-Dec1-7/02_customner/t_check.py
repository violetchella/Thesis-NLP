import spacy

from spacy import displacy
nlp = spacy.load('output/model-best') 
text="Among whom was Mary Magdalen and Mary the mother of James and Joseph and the mother of the sons of Zebedee"
view=nlp(text)
#visualize
for ent in view.ents:
      print(ent.text, ent.start_char, ent.end_char, ent.label_)
