import numpy as np
import pandas as pd
import texthero as hero
from texthero import preprocessing as ppe
import sklearn
import xgboost
import pickle

if __name__=="__main__":
    df = pd.read_csv("kaggle_fake_train.csv")
    df.drop(['id','title','author'],axis=1,inplace=True)
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
    df['cleaned_text'] = hero.clean(df['text'], custom_pipeline)
    vector  = hero.CountVectorizer()
    x = vector.fit_transform(df['cleaned_text'])
    y = df['label'].values
    
    from sklearn.model_selection import train_test_split
    #train_test_split
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 69)
    
    #model 
    from xgboost import XGBClassifier
    xgb = XGBClassifier() 
    xgb.fit(X_train,y_train)
    pred = xgb.predict(X_test)
    print(sklearn.metrics.accuracy_score(y_test, pred))
    
    #saving models
    pickle.dump(vector, open('tfidfvect.pkl', 'wb'))
    pickle.dump(xgb, open('model.pkl', 'wb'))
    

