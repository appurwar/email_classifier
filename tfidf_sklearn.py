#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 17:03:36 2017

@author: apoorv
"""


import nltk
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn import svm, tree
from sklearn.neural_network import MLPClassifier

import glob
import errno
import string
import re
from gensim.sklearn_api.w2vmodel import W2VTransformer
import csv
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer


def preprocess(file_list):
    flat_list = list();
    raw_documents = []
    translator = str.maketrans('', '', string.punctuation)
    porter = nltk.PorterStemmer()
    wnl = nltk.WordNetLemmatizer()
    
    for name in file_list: # 'file' is a builtin type, 'name' is a less-ambiguous variable name.
        try:
           with open(name,'rb') as f:
            flat_list = []
            document = ''
            for line in f:
                line = line.strip()
                line = str(line)
                split = line.split()
                
                if split:              
                    flat_list = line.split()
                    long_words = [w for w in flat_list if len(w) > 2]
                    
                    long_words_noPunc = [w.translate(translator) for w in long_words]
                   
                    
                    long_words_lower = [x.lower() for x in long_words_noPunc]
                    
                    stops = set(stopwords.words("english"))   
                    meaningful_words = [w for w in long_words_lower if not w in stops]  
                   # meaningful_words = ' '.join(meaningful_words)
                    
                   
                    long_word_stem = [porter.stem(t) for t in meaningful_words]
                   # long_word_stem = ' '.join(long_word_stem)
                   # document = document + ' '  + long_word_stem
                    long_word_lem = [wnl.lemmatize(t) for t in long_word_stem]
                     
                    long_word_lem = ' '.join(long_word_lem)
                    document = document + ' '  + long_word_lem
                    
            raw_documents.append(document)
    #                print(long_word_lem)
        except IOError as exc:
            if exc.errno != errno.EISDIR: # Do not fail if a directory is found, just ignore it.
                raise # Propagate other kinds of IOError.
    return raw_documents



spam_train_path = '/Users/apoorv/MyDocs/CU Courses/Fall 2017/ML/HW/HW3/email_classification_data/train_data/spam/*.txt'
ham_train_path = '/Users/apoorv/MyDocs/CU Courses/Fall 2017/ML/HW/HW3/email_classification_data/train_data/ham/*.txt'   

spam_train_files = glob.glob(spam_train_path)
ham_train_files = glob.glob(ham_train_path)   

all_test_files = []
spam_test_files = spam_train_files[0:50]
ham_test_files = ham_train_files[0:50]

all_test_files.extend(spam_test_files)
all_test_files.extend(ham_test_files)


ham_train_files = ham_train_files[50:]
spam_train_files = spam_train_files[50:]


all_train_files = []
all_train_files.extend(spam_train_files)
all_train_files.extend(ham_train_files)



#0 - ham
#1 - spam
train_labels = []
train_labels.extend([1] * len(spam_train_files))
train_labels.extend([0] * len(ham_train_files))


test_labels = []
test_labels.extend([1] * len(spam_test_files))
test_labels.extend([0] * len(ham_test_files))





#==============================================================================
# w2v = W2VTransformer(size=400, min_count=3)
# X_train_w2v = w2v.fit_transform(raw_train_documents,train_labels)
# print(X_train_w2v.shape)
#==============================================================================

raw_train_documents = preprocess(all_train_files)

#tfidf = TfidfVectorizer(input='content')



vectorizer = CountVectorizer(analyzer = "word",   
                                tokenizer = None,    
                                preprocessor = None, 
                                stop_words = None,   
                                max_features = 6000) 



X_train_tfidf = vectorizer.fit_transform(raw_train_documents)
print(X_train_tfidf.shape)

test_path = '/Users/apoorv/MyDocs/CU Courses/Fall 2017/ML/HW/HW3/email_classification_data/test_data/*.txt'
test_files = glob.glob(test_path)
raw_test_documents = preprocess(all_test_files)
X_test_tfidf = vectorizer.transform(raw_test_documents)
print(X_test_tfidf.shape)


#clf = MultinomialNB().fit(X_train_tfidf, train_labels)
#predictions = clf.predict(X_test_tfidf)

#clf = svm.SVC(C = 0.1, kernel='poly', degree=10)
#clf.fit(X_train_tfidf, train_labels)

#clf = tree.DecisionTreeClassifier()
#clf.fit(X_train_tfidf, train_labels)

clf = MLPClassifier(activation='relu', solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(90,2), random_state=1)
clf.fit(X_train_tfidf, train_labels)

predictions = clf.predict(X_test_tfidf)
print(accuracy_score(predictions, test_labels))

