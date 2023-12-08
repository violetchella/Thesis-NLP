import pandas as pd
from tqdm import tqdm
import spacy
from spacy.tokens import DocBin
import json

def load_data(file):
    with open(file,'r',encoding='utf-8') as f:
        data=json.load(f)
    return(data)    



TRAIN_DATA= load_data("D:/VScode-py/nlp/annotations.json")


nlp = spacy.blank("en") # load a new spacy model
db = DocBin() # create a DocBin object

for text, annot in tqdm(TRAIN_DATA): # data in previous format
    doc = nlp.make_doc(text) # create doc object from text
    ents = []
    for start, end, label in annot["entities"]: # add character indexes
        span = doc.char_span(start, end, label=label, alignment_mode="contract")
        if span is None:
            print("Skipping entity")
        else:
            ents.append(span)
    doc.ents = ents # label the text with the ents
    db.add(doc)
db.to_disk("./train.spacy") # save the docbin object