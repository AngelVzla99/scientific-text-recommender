# paths
main_path = 'drive/MyDrive/Universidad/Tesis_sistema_de_recomendacion'
path_to_embeddings = main_path+'/Embeddings'
path_to_dataset = main_path+'/Dataset'
path_to_test_set = main_path+'/Conjuntos_de_prueba'
path_to_results = main_path+'/Resultados'
path_to_MLP = main_path+'/MLP_models'
path_to_embeddings_graph = main_path+'/Embeddings_graph'

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
from gatem import gatem
from cf import collaborative_filtering_recommender
from mf import matrix_factorization_recommender

class cc_recommender():
  # cc similarity
  dois = []
  citation_inv = None
  pos = None
  # Parameters 
  alpha = None

  def cc_similarity( self, doi ):
    # Microsoft cc matrix
    pos_doi = self.pos[doi]
    ans_row = np.zeros( len(self.dois) )
    for doi1 in self.dois:
      if doi1 in self.citation_inv:
        papers = self.citation_inv[doi1]
        for i in range(len(papers)):
          for j in range(i+1,len(papers)):
            p1 = self.pos[papers[i]]
            p2 = self.pos[papers[j]]
            if p1 == pos_doi: ans_row[p2] += 1
            if p2 == pos_doi: ans_row[p1] += 1
    ans_sort = [ [ans_row[i],self.dois[i]] for i in range(len(ans_row)) ]
    return self.sort_and_split( ans_sort )

  def sigmoid(self,x):
    # Values after ignore the 0's in the matrix
    tita = 0.473749 # (paper) 0.4 , 0.6 
    # tita = std
    tau = 1.12816 # (paper) 5.0 , 0.3 
    # Tau = promedio de cc
    return 1 / (1 +  np.exp(tita * ( tau - x ) ))

  def __init__(self, alpha = 1.5):
    self.alpha = float(alpha)
    # Let's create citation matrix
    references = pd.read_csv(path_to_dataset+'/Reference.csv')
    nodes = {}
    self.citation_inv = {}
    for index, row in references.iterrows():
      # Save nodes
      nodes[row['from']] = True
      nodes[row['to']] = True 
      # Save inverse citation
      if not row['to'] in self.citation_inv:
        self.citation_inv[row['to']] = []
      self.citation_inv[ row['to'] ].append( row['from'] )

    df_papers = pd.read_csv(path_to_dataset+'/Paper.csv') 
    self.dois = df_papers['paper_id'].to_list()

    # Map dois to integer
    self.pos = {}
    index = 0
    for doi in self.dois: 
      self.pos[doi] = index
      index += 1
    del references

  def get_recommendation_V2( self, doi ):
    ids, values = self.cc_similarity(doi) 
    values = [ self.alpha*self.sigmoid(x) for x in values ]
    return ids, values
  
class hybrid_recommender():
  # Models
  system_CC = None
  system_EM = None
  system_CF = None
  system_GATEM = None
  # Alphas
  alpha_GATEM = None
  
  def sort_and_split(self, results):
    # retult = [ [similarity,doi] ]
    # Sort results    
    results.sort(); results.reverse()
    # Answers 
    ans_dois = [ x[1] for x in results ]
    ans_similarity = [ x[0] for x in results ]
    return ans_dois, ans_similarity

  def __init__(self, embedder_name, use_CF=False, alpha = 1.5, gatem_alpha=None):
    # Load cc recommender
    if alpha!=None:
      self.system = cc_recommender( alpha )
      print("CC matrix ready")
    # Load embedder recommender
    if embedder_name!=None:
      self.system_EM = emb_recommender(embedder_name)
      print("Embedder ready")
    # Load colaborative filering recommender
    if use_CF!=None and use_CF==True:
      self.system_CF = collaborative_filtering_recommender(True)
      print("Collaborative filtering ready")
    # Load gatem recommender
    if gatem_alpha!=None:
      self.system_GATEM = gatem('model_V6_authors_FF_CLS_TA', 'scincl_V3_CLS_title&abstract', 'big_graph')
      self.alpha_GATEM = gatem_alpha
      print("GATEM ready")

  def get_recommendation(self, doi):
    return []    
    # Get recommendation from embeddings
    ids, values = self.system.recommendation_by_id_threshold(doi,0) 
    # Get recommendation from cc matrix
    ids2, values2 = self.cc_similarity(doi) 
    values2 = [ self.alpha*self.sigmoid(x) for x in values2 ]

    ans = {}
    for i in range(len(ids)): ans[ids[i]] = values[i]
    for i in range(len(ids2)): 
      if ids2[i] in ans:
        ans[ids2[i]] = values2[i] if values2[i] > ans[ids2[i]] else ans[ids2[i]]
      else: 
        ans[ids2[i]] = values2[i]
    
    # ARRELGAR ESTO ANGEL
    key_value = [ [ ans[x], x ] for x in ans.keys() ]
    return self.sort_and_split( key_value )
  
  def get_recommendation_V2(self, doi):
    # Get recommendation from cc matrix
    ids, values = self.system_EM.get_recommendation_V2(doi)

    list_of_systems = []
    if self.system_CC!=None: list_of_systems.append( (1,self.system_CC) )
    if self.system_CF!=None: list_of_systems.append( (1,self.system_CF) )
    if self.system_GATEM!=None: list_of_systems.append( (self.alpha_GATEM,self.system_GATEM) )

    # Default values, cc recommendation
    ans = {}
    for i in range(len(ids)): ans[ids[i]] = values[i]
    # Add new systems
    for alpha, system in list_of_systems:
      # Ger recommendation of the system
      ids2, values2 = system.get_recommendation_V2(doi)
      values2 = [ alpha*x for x in values2 ]
      # Maximum between recommendations
      for i in range(len(ids2)): 
        if ids2[i] in ans:
          ans[ids2[i]] = values2[i] if values2[i] > ans[ids2[i]] else ans[ids2[i]]
        else: 
          ans[ids2[i]] = values2[i]  

    relevance = [ ans[x] for x in ans.keys() ]
    dois = [ x for x in ans.keys() ]
    return dois, relevance