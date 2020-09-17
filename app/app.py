from flask import Flask,render_template,url_for,request
import numpy as np
import pandas as pd
import re

from pathlib import Path
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    TextClassificationPipeline
)

model_loc = "troll_detect"

nlp_clf = TextClassificationPipeline(
    model=DistilBertForSequenceClassification.from_pretrained(model_loc),
    tokenizer=DistilBertTokenizerFast.from_pretrained(model_loc)
)

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\'t", " not", text) # Change 't to 'not'
    text = re.sub(r'(@.*?)[\s]', ' ', text) # Remove @name
    text = re.sub(r"$\d+\W+|\b\d+\b|\W+\d+$", " ", text) # remove digits
    text = re.sub(r"[^\w\s\#]", "", text) #remove special characters except hashtags
    text = text.strip(" ")
    text = re.sub(' +',' ', text).strip() # get rid of multiple spaces and replace with a single
    return text

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        message = clean_text(message)
        data = [message]
        my_prediction = nlp_clf(data)[0]['label']
    return render_template('result.html', prediction = my_prediction)

if __name__ == '__main__':
	app.run(debug=True)
