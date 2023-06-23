# TF IDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# doc2vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
# Glove pretrained with twitter
import gensim.downloader as apiGlove
import numpy as np
# Bert with tensor flow (and keras)
import os
import shutil
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization  # to create AdamW optimizer
import matplotlib.pyplot as plt
tf.get_logger().setLevel('ERROR')
# Specter
from transformers import AutoTokenizer, AutoModel
from transformers import AutoTokenizer, TFAutoModel

glove_model = apiGlove.load('glove-twitter-25')

bert_preprocessor = hub.KerasLayer(
    "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
bert_encoder = hub.KerasLayer(
    "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4",
    trainable=True)

# load model and tokenizer (specter)
tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
model = AutoModel.from_pretrained('allenai/specter').to("cuda:0")

# load model and tokenizer (scincl)
tokenizer_scincl = AutoTokenizer.from_pretrained('malteos/scincl')
model_scincl = AutoModel.from_pretrained('malteos/scincl').to("cuda:0")
#model_scincl = TFAutoModel.from_pretrained('malteos/scincl',from_pt=True)

# load model and tokenizer (bert large)
from transformers import BertTokenizer, BertModel
tokenizer_bert_large = BertTokenizer.from_pretrained('bert-large-cased')
#model_bert_large = TFBertModel.from_pretrained("bert-large-cased")
model_bert_large = BertModel.from_pretrained("bert-large-cased").to("cuda:0")