
# FAKE NEWS CLASSIFIER

Fake News classifier is a Machine learning Model which has been trained using kaggle dataset and classifies  whether the given news is fake or real with overall accuracy of 91%


## DATA
https://github.com/saiabhiteja/Fakenews-Classifier/tree/main/data

## Models used ,their Accuracies
| Models                  | Accuracy |
|-------------------------|------------|
| Logistic Regression     | 87.4%      |
| Multinomial Navie bayes | 88.6%      |
| xgboost                 | 91%        |

## Project Tree Structure
```
📦 Fakenews-Classifier
├── Procfile 
├── README.md
├── app.py
├── data
     ├── kaggle_fake_test.7z
     ├── kaggle_fake_train.7z
├── Model.pkl
├── Model.py
├── requirements.txt
├── static
     ├── css
        ├── news2.jpg
        ├── style.css
├── templates
      ├── home.html
└── tfidfvect.pkl
```

## Tools Used
- Programming language : Python

- Visualization : Matplotlib and Seaborn

- Front end development : HTML/CSS

- Back end development : Flask

- Version control system : GitHub

- Preprocessing : TextHero

- Deployment : Heroku

## References
 [Text Hero ](https://texthero.org/)
 
[Xgb classifier](https://towardsdatascience.com/beginners-guide-to-xgboost-for-classification-problems-50f75aac5390)

[Tfidf Vectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)

[Heroku](https://devcenter.heroku.com/articles/getting-started-with-python)
