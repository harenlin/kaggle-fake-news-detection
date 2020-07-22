import pandas as pd
import numpy as numpy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def makecsv(id_test, test_y_pred):
    test_y_pred = test_y_pred
    submission = {'id': id_test, 'label': test_y_pred}
    submission = pd.DataFrame(submission)
    submission.to_csv('submission.csv', index=0, header=1)
    
# read data
train_corpus = pd.read_csv('./train.csv')
test_corpus = pd.read_csv('./test.csv')

# training data
x_train = train_corpus['text'].fillna('0')
y_train = train_corpus['label']

# testing data
x_test = test_corpus['text'].fillna('0')
id_test = test_corpus['id']

# TF
TF_vectorizer = CountVectorizer(stop_words='english')
train_X_TF    = TF_vectorizer.fit_transform( x_train )
test_X_TF     = TF_vectorizer.transform( x_test )

# logistic regression
print('TF + logistic regression')
classifier = LogisticRegression().fit( train_X_TF, y_train.astype('int') )
# ".astype('int')" deal with ValueError: Unknown label type: 'unknown'
print("the score of the model is "
       +str(classifier.score( train_X_TF, y_train.astype('int'))))
pred_TF = classifier.predict(test_X_TF)
print( pred_TF )

# make submission file
makecsv(id_test, pred_TF)
