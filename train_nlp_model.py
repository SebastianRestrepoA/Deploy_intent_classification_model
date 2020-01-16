import os
import warnings
import numpy as np
from text_processing_fns import *
warnings.filterwarnings("ignore")
from classifiers import *
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import pickle
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

" LOAD DATA BASE"
vPathKnowledgeBase = './KnowledgeBase.xlsx'
#
KnowledgeBase = pd.read_excel(vPathKnowledgeBase)

# label encode the target variable to transform non-numerical labels
encoder = preprocessing.LabelEncoder()
y = encoder.fit_transform(KnowledgeBase["Intent"])  # numerical labels
intent_names = encoder.classes_
nfolds = 10

" NATURAL LANGUAJE PROCESSING"

# transform our text information in lowercase
# KnowledgeBase["Utterance"] = lowercase_transform(KnowledgeBase["Utterance"])
#
# # Removing punctuation characters such as: !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
# KnowledgeBase["Utterance"] = remove_characters(KnowledgeBase["Utterance"])
#
# # Removing stop words from text
# KnowledgeBase["Utterance"] = remove_stopwords(KnowledgeBase["Utterance"])

" WORD2VECT TRANSFORM AND FEATURE EXTRACTION"

features_matrix = feature_engineering(KnowledgeBase["Utterance"], tf_idf=True)


" MODEL DEVELOPMENT"

x = features_matrix['TF-IDF']['matrix']
vocabulary_object = features_matrix['TF-IDF']['object']
svm_model = fn_search_best_svm_classifier(x, y, nfolds, 'TF-IDF', display_results=True)

pickle.dump(svm_model, open('nlp_model.pkl', 'wb'))
pickle.dump(vocabulary_object, open('knowledgebase_vocabulary.pkl', 'wb'))
pickle.dump(intent_names, open('intent_names.pkl', 'wb'))
