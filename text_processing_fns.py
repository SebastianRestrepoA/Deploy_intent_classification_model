import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from matplotlib import rcParams
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import unicodedata
rcParams.update({'figure.autolayout': True})


def count_words(vKnowledgeBase):

    """ This function computes the number of words in each utterance belonging to the knowledge base.

    :param vKnowledgeBase: pandas series with the utterances of the knowledge base.

    :return: pandas series with the number of words used in each utterance.


    """
    return vKnowledgeBase.apply(lambda x: len(str(x).split()))


def remove_stopwords(vKnowledgeBase):

    """ This function eliminates the stop words from each utterance belonging to the knowledge base.

    :param vKnowledgeBase: pandas series with the utterances of the knowledge base.

    :return: pandas series with the knowledge base without stop words.


    """
    stop = stopwords.words('spanish')
    return vKnowledgeBase.apply(lambda x: " ".join(x for x in x.split() if x not in stop))


def remove_tilde(vUtterance):

    """ This function removes the accent mark (or "tilde") from a utterance.

    :param vUtterance: string variable with a utterance.

    :return: string variable with the utterance without "tildes".


    """

    return ''.join((c for c in unicodedata.normalize('NFD', vUtterance) if unicodedata.category(c) != 'Mn'))


def remove_characters(vKnowledgeBase):

    """ This function removes the irrelevant puntuation characters (!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~)
    from each utterance belonging to the knowledge base.

    :param vKnowledgeBase: pandas series with the utterances of the knowledge base.

    :return: pandas series with the knowledge base without irrelevant characters.


    """
    return vKnowledgeBase.str.replace('[^\w\s]', '')


def lowercase_transform(vKnowledgeBase):

    """ This function transforms the utterances belonging to the knowledge base to lowercase.

    :param vKnowledgeBase: pandas series with the utterances of the knowledge base.

    :return: pandas series with the knowledge base transformed to lowercase.


    """

    return vKnowledgeBase.apply(lambda x: " ".join(x.strip().lower() for x in x.split()))


def text_lemmatization(utterance):

    """ This function apply lemmatization over all the words of a utterance.

    :param utterance: string variable.

    :return: string variable lemmatized.

    """

    words = word_tokenize(utterance)
    lemmatizer = WordNetLemmatizer()
    words_lemma = [lemmatizer.lemmatize(word) for word in words]

    return ' '.join(words_lemma)


def lemmatization_transform(vKnowledgeBase):

    """ This function apply lemmatization in every utterance belonging to the knowledge base.

    :param vKnowledgeBase: pandas series with the utterances of the knowledge base.

    :return: pandas series with lemmatized knowledge base.


    """

    return vKnowledgeBase.apply(lambda x: text_lemmatization(x))


def stemming_transform(vKnowledgeBase):

    """ This function apply stemming function in every utterance belonging to the knowledge base.

    :param vKnowledgeBase: pandas series with the utterances of the knowledge base.

    :return: pandas series with stemmed knowledge base.

    """

    return vKnowledgeBase.apply(lambda x: text_stemming(x))


def text_stemming(utterance):

    """ This function apply stemming over all the words of a utterance.

    :param utterance: string variable.

    :return: stemmed string variable.

    """

    stemmer = SnowballStemmer("spanish")
    words = word_tokenize(utterance)
    words_stem = [stemmer.stem(word) for word in words]

    return ' '.join(words_stem)


def feature_engineering(vUtterances, vectorizer=False, tf_idf=False, ngram=False):

    """ This function coverts utterances into feature matrices such as word2vect, TF-IDF and n-gram for training
    of ML models.

    :param vUtterances: pandas series with the utterances of knowledge base.

    :return: dictionary with 3 features matrices corresponding to the count vector, TF-IDF and n-grams transforms.

    """

    features = {}
    vUtterances.index = range(0, len(vUtterances))

    if vectorizer is True:

        # transform the training data using count vectorizer object
        # create a count vectorizer object
        count_vect = CountVectorizer(analyzer='word')
        count_vect.fit(vUtterances) # Create a vocabulary from all utterances
        x_count = count_vect.transform(vUtterances)  # Count how many times is each word from each utterance in the
        # vocabulary.
        features['count_vectorizer'] = {'object': count_vect, 'matrix': x_count}
        # pd.DataFrame(x_count.toarray(), columns=count_vect.get_feature_names())

    if tf_idf is True:

        # word level tf-idf

        " TF-IDF score represents the relative importance of a term in the document and the entire corpus. "
        tfidf_vect = TfidfVectorizer(analyzer='word', max_features=5000)  # token_pattern=r'\w{1,}'
        tfidf_vect.fit(vUtterances)
        x_tfidf = tfidf_vect.transform(vUtterances)
        features['TF-IDF'] = {'object': tfidf_vect, 'matrix': x_tfidf}

    if ngram is True:

        # ngram level tf-idf
        tfidf_vect_ngram = TfidfVectorizer(analyzer='word', ngram_range=(2, 3), max_features=5000) # token_pattern=r'\w{1,}'
        tfidf_vect_ngram.fit(vUtterances)
        x_tfidf_ngram = tfidf_vect_ngram.transform(vUtterances)
        features['ngram'] = {'object': tfidf_vect_ngram, 'matrix': x_tfidf_ngram}

    return features






