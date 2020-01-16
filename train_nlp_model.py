import warnings
from text_processing_fns import *
from classifiers import *
import pandas as pd
import pickle
warnings.filterwarnings("ignore")


" LOAD DATA BASE"
vPathKnowledgeBase = './KnowledgeBase.xlsx'
#
KnowledgeBase = pd.read_excel(vPathKnowledgeBase)

# label encode the target variable to transform non-numerical labels
encoder = preprocessing.LabelEncoder()
y = encoder.fit_transform(KnowledgeBase["Intent"])  # numerical labels
intent_names = encoder.classes_
nfolds = 10


" WORD2VECT TRANSFORM AND FEATURE EXTRACTION"

features_matrix = feature_engineering(KnowledgeBase["Utterance"], tf_idf=True)


" MODEL TRAINING AND DEVELOPMENT"

x = features_matrix['TF-IDF']['matrix']
vocabulary_object = features_matrix['TF-IDF']['object']
svm_model = fn_search_best_svm_classifier(x, y, nfolds, 'TF-IDF', display_results=True)


"SAVE FINAL MODEL"
pickle.dump(svm_model, open('nlp_model.pkl', 'wb'))
pickle.dump(vocabulary_object, open('knowledgebase_vocabulary.pkl', 'wb'))
pickle.dump(intent_names, open('intent_names.pkl', 'wb'))
