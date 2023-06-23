# paths
main_path = 'drive/MyDrive/Universidad/Tesis_sistema_de_recomendacion'
path_to_embeddings = main_path+'/Embeddings'
path_to_dataset = main_path+'/Dataset'
path_to_test_set = main_path+'/Conjuntos_de_prueba'
path_to_results = main_path+'/Resultados'

# Others
import ast
import numpy as np
import pandas as pd
import math
# This version of embedding_factory only have TFIDF
# TF IDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# Lematization
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
nltk.download('omw-1.4')
import re
# Data manipulation 
import os
import sys
# Metrics
from sklearn.metrics import ndcg_score

class collaborative_filtering_recommender():
  behavior = None
  dois = None

  def similarity( self, behavior1, behavior2 ):
    # intersection
    inte = set(behavior1).intersection(behavior2)
    x11 = len(inte)
    x10 = len(behavior1) - x11
    x01 = len(behavior2) - x11
    if( x11+x10+x01 == 0 ): return 0
    return x11/(x11+x10+x01)

  def __init__(self,add_citations_str):
    # Search useful behavior of the users
    bahavior_matrix = pd.read_csv( path_to_test_set + "/collaborative_filtering.csv" )
    # Convert to dictionary
    self.behavior = {}
    for index,row in bahavior_matrix.iterrows():
      # save as dictionary
      self.behavior[row['from']] = row['to'].split(";")
    # save all paper dois
    self.dois = pd.read_csv(path_to_dataset+"/Paper.csv")["paper_id"].to_list()
    for doi in self.dois: 
      if not doi in self.behavior: 
        self.behavior[doi] = []
    # Add citations to the behavior adyacent list
    if add_citations_str=="True":
      citations = pd.read_csv( path_to_dataset + '/Reference.csv' )
      for index, row in citations.iterrows():
        if row['to'] not in self.behavior[row['from']]:
          self.behavior[row['from']].append( row['to'] )        
        
  def get_recommendation_V2(self, target):
    paper_ids = [None]*(len(self.dois))
    relevance = [None]*(len(self.dois))
    for index, doi in enumerate(self.dois):      
      paper_ids[index] = doi
      relevance[index] = self.similarity( self.behavior[target], self.behavior[doi] )
    return paper_ids, relevance