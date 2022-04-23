import numpy as np
import nltk
import unidecode
# nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
from underthesea import word_tokenize
stemmer = PorterStemmer()

def tokenize(sentence):
    """
    split sentence into array of words/tokens
    a token can be a word or punctuation character, or number
    """
    # word_tokenize in nltk.tokenize.punkt use to tokenize sentence
    return nltk.word_tokenize(sentence) 

def tokenizeVN(sentence):
    """
    split sentence into array of words/tokens
    a token can be a word or punctuation character, or number
    """
    # word_tokenize in nltk.tokenize.punkt use to tokenize sentence
    word_orgin =  word_tokenize(sentence)
    # print (word_orgin)
    result = []
    for w in word_orgin:
        result.append(unidecode.unidecode(w).replace(" ",""))        
    return result


def stem(word):
    """
    stemming = find the root form of the word
    examples:
    words = ["organize", "organizes", "organizing"]
    words = [stem(w) for w in words]
    -> ["organ", "organ", "organ"]
    """
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, words):
    """
    return bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise
    example:
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bog   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """
    # stem each word
    sentence_words = [stem(word) for word in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag