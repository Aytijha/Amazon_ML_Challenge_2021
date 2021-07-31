import pandas as pd
import numpy as np
import csv
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.metrics import accuracy_score

#test file
df_test = pd.read_csv('test.csv', escapechar = "\\", quoting = csv.QUOTE_NONE)
print("Test Head")
#print(df_test.head())

'''#test columns
for col in df_test.columns:
  print(col)'''

#train file
df_train = pd.read_csv('train.csv', escapechar = "\\", quoting = csv.QUOTE_NONE)
print("Train Head")
#print(df_train.head())

'''#train columns
for col in df_train.columns:
  print(col)'''

print("Train.isna().sum():")
#print(df_train.isna().sum())

print("Train Shape:")
#print(df_train.shape)

'''print(df_train['TITLE'].is_unique)
print(df_train['DESCRIPTION'].is_unique)
print(df_train['BULLET_POINTS'].is_unique)
print(df_train['BRAND'].is_unique)
print(df_train['BROWSE_NODE_ID'].is_unique)'''

df = df_train.dropna()
print("New Train.isna().sum():")
#print(df.isna().sum())

punctuation_signs = list("?:!.,;")
nltk.download('punkt')
nltk.download('wordnet')
wordnet_lemmatizer = WordNetLemmatizer()
nltk.download('stopwords')
stop_words = list(stopwords.words('english'))

#Cleaning TITLE
print("Cleaning TITLE column:")

print("Removing escape sequence characters...")
df['Title'] = df['TITLE'].str.replace("\r", " ")
df['Title'] = df['Title'].str.replace("\n", " ")
df['Title'] = df['Title'].str.replace("    ", " ")
df['Title'] = df['Title'].str.replace('"', '')
df['Title'] = df['Title'].str.lower()
for punct_sign in punctuation_signs:
  df['Title'] = df['Title'].str.replace(punct_sign, '')
df['Title'] = df['Title'].str.replace("'s", "")

print("Lemmatizing Data...")
nrows = len(df)
lemmatized_text_list = []
for row in range(0, nrows):
    lemmatized_list = []
    text = df.iloc[row, 5]
    text_words = text.split(" ")
    for word in text_words:
        lemmatized_list.append(wordnet_lemmatizer.lemmatize(word, pos="v"))
    lemmatized_text = " ".join(lemmatized_list)
    lemmatized_text_list.append(lemmatized_text)
df['Title'] = lemmatized_text_list

print("hehe")

print("Dropping unnecessary columns...")
final_cols = ["Title", "BROWSE_NODE_ID"]
df = df[final_cols]
df = df.iloc[:15000, :]
print("New Dataframe head:")
#print(df.head())

print("Performing Train-Test-Split...")
X_train, X_test, y_train, y_test = train_test_split(df["Title"],
                                                    df["BROWSE_NODE_ID"],
                                                    test_size=0.15, 
                                                    random_state=8)

'''ngram_range = (1,2)
min_df = 10
max_df = 1.
max_features = 300'''

print("Vectorizing...")
tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1,2), min_df=5)
print('1')
X_train_vectors_tfidf = tfidf.fit_transform(X_train).toarray()
#labels_train = y_train
print(X_train_vectors_tfidf.shape)
print('1')
X_test_vectors_tfidf = tfidf.transform(X_test).toarray()
#labels_test = y_test
print(X_test_vectors_tfidf.shape)

print('LogReg...')
lr_tfidf=LogisticRegression(solver = 'liblinear', C=10, penalty = 'l2')
lr_tfidf.fit(X_train_vectors_tfidf, y_train)

print("Prediction...")
y_predict = lr_tfidf.predict(X_test_vectors_tfidf)
y_prob = lr_tfidf.predict_proba(X_test_vectors_tfidf)[:,1]
#print(classification_report(y_test,y_predict))
print('Confusion Matrix:',confusion_matrix(y_test, y_predict))

print(accuracy_score(y_test, y_predict))
