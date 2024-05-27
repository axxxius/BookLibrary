import numpy as np
import nltk

nltk.download('punkt')
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords

stemmer = PorterStemmer()
snowball = SnowballStemmer(language="russian")


def tokenize(sentence):
    return nltk.word_tokenize(sentence, language="russian")


def stop_words(tokens):
    filtered_tokens = []
    stop_words = stopwords.words("russian")
    for token in tokens:
        if token not in stop_words:
            filtered_tokens.append(token)
    return filtered_tokens


def stem(word):
    return snowball.stem(word.lower())


def bag_of_words(tokenized_sentence, words):
    # stem each word
    sentence_words = [stem(word) for word in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1
    return bag
