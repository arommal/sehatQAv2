import re
import numpy as np
import pickle
from nlp_id.lemmatizer import Lemmatizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import gensim
from gensim.models import fasttext, FastText, KeyedVectors

def clean_text(text):
    '''
    Apply text preprocessing:
    1. Case folding into lowercase
    2. Delete punctuation
    3. Delete newlines
    4. Delete numbers and symbols
    5. Lemmatize
    6. Delete space duplicates
    7. Delete stopwords
    '''
    lemmatizer = Lemmatizer()
    stop_factory = StopWordRemoverFactory()
    more_stopwords = ['halo', 'dengan', 'ia', 'bahwa', 'oleh', 'nya', 'jadi', 'untuk', 'dok', 'dokter', 'assalamualaikum', 'selamat', 'terimakasih']
    stopwords = stop_factory.get_stop_words() + more_stopwords
    
    text = text.strip().lower()
    text = re.sub(r'(\W)(?=\1)', '', text) #hapus duplikat tanda baca
    text = text.replace("\n", " ") #hapus line break
    sentence = re.sub(r'[^a-zA-Z]', ' ', text) #hapus simbol dan angka
    sentence = re.sub(r"\b[a-zA-Z]\b", " ", sentence) #remove 1 alphabet
    sentence = lemmatizer.lemmatize(sentence) # lemmatisasi
    sentence = re.sub(' +', ' ', sentence) # remove double space
    sentence = sentence.split(" ")
    sentence = [word for word in sentence if word not in stopwords] # hapus stopword
    result_sentence = " ".join(sentence).strip()
    
    return result_sentence


def load_tokenizer():
    with open('tokenizer/tokenizer_text_fixtypo.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    return tokenizer


def tokenize(text, tokenizer):
    '''
    Tokenize cleaned question. Tokenizer is trained on the whole dataset question and answer texts and saved as a pickle file.
    '''
    sequences = tokenizer.texts_to_sequences([text])

    MAX_LENGTH = 300
    padded_sequences = pad_sequences(sequences, maxlen=MAX_LENGTH, padding='post')
    
    return padded_sequences


def load_fasttext_vocab():
    with open('fasttext_word_dict.pickle', 'rb') as handle:
        vocab = pickle.load(handle)

    return vocab


def embed_for_similarity(s, ft):
    stop_factory = StopWordRemoverFactory()
    more_stopwords = ['halo', 'dengan', 'ia', 'bahwa', 'oleh', 'nya', 'jadi', 'untuk', 'dok', 'dokter', 'assalamualaikum', 'selamat', 'terimakasih']
    stopwords = stop_factory.get_stop_words() + more_stopwords
    a = []
    for word in s.split():
        try:
            if word not in stopwords:
                a.append(ft[word])
        except KeyError:
            continue

    if len(a) == 0:
        a.append([0] * 300)

    s_emb = np.mean(a, axis=0)

    return s_emb