#importing the libraries
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import texthero as hero
from texthero import preprocessing as ppe
import sklearn
import xgboost


app = Flask(__name__)
vector = pickle.load(open('tfidfvect.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

def clean_text(s):
    custom_pipeline = [ppe.fillna,
                   ppe.lowercase,
                   ppe.remove_punctuation,
                   ppe.remove_stopwords,
                   ppe.remove_digits,
                   ppe.remove_diacritics,
                   ppe.remove_round_brackets,
                   ppe.remove_html_tags,
                   ppe.remove_urls,
                   ppe.remove_whitespace,
                   ppe.remove_brackets,
                   ppe.remove_angle_brackets,
                   ppe.remove_curly_brackets,
                   ppe.remove_square_brackets,
                   ppe.stem
                  ]
    df = pd.DataFrame(data={
    'text':[s]},)
    d = hero.clean(df['text'], custom_pipeline)
    return d[0]
    

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    f = request.form.values()
    c = clean_text(f)
    dfd = pd.DataFrame(data={
    'text':[c]},)
    x = vector.transform(dfd['text'])
    prediction = model.predict(x)
    
    
    if prediction==1:
        output = "Real"
    else:
        output = "Fake"


    return render_template('home.html', prediction_text='News is {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)

