# paths
main_path = 'drive/MyDrive/Universidad/Tesis_sistema_de_recomendacion'
path_to_embeddings = main_path+'/Embeddings'
path_to_embeddings_graph = main_path+'/Embeddings_graph'
path_to_dataset = main_path+'/Dataset'
path_to_test_set = main_path+'/Conjuntos_de_prueba'
path_to_results = main_path+'/Resultados'

# TF IDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# Others
import ast
import numpy as np
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
import embedding_factory
import embedding_manager

class emb_recommender():
  matrix = [] # Cosine similarity matrix of the papers
  papers = [] # Papers with the embeddings
  path_papers = path_to_dataset+'/Paper.csv'
  position = None # Position of the articles in the matrix

  embeddings = []

  # ===========================
  #    Auxiliar functions    //
  # ===========================

  def get_by_threshold( self, thre, papers, M, pos_paper ):
    '''
    M is the cosine similarity matrix
    '''
    # Top n articles
    n = len(M[0]) if type(M[0])==list else M.shape[1]
    ans = []
    for j in range(n):
      if j!=pos_paper and M[pos_paper][j]>=thre:
        ans.append([ M[pos_paper][j], j ])
    ans.sort(); ans.reverse()
    # Sorting    
    top_n_ids = [ papers.iloc[ans[i][1]]['paper_id'] for i in range(len(ans)) ]
    top_n_values = [ ans[i][0] for i in range(len(ans)) ]
    return top_n_ids, top_n_values

  def get_top_n( self, N, papers, M, pos_paper ):
    '''
    M is the cosine similarity matrix
    '''
    # Top n articles
    n = len(M[0]) if type(M[0])==list else M.shape[1]
    ans = []
    for j in range(n):
      if j!=pos_paper:
        ans.append([ M[pos_paper][j], j ])
    # Sorting
    ans.sort(); ans.reverse()
    max_n = N if N < len(ans) else len(ans)
    top_n_ids = [ papers.iloc[ans[i][1]]['paper_id'] for i in range(max_n) ]
    top_n_values = [ ans[i][0] for i in range(max_n) ]
    return top_n_ids, top_n_values

  def get_matrix_and_papers( self, model_name ):
    em = embedding_manager()
    papers_model = em.load_and_join(model_name)
    list_of_embeddings = papers_model['embedding'].tolist() 
    embeddings_model = em.get_numpy_embedding(list_of_embeddings)  
    M_model = cosine_similarity( embeddings_model, embeddings_model )

    self.embeddings = embeddings_model

    del list_of_embeddings
    del embeddings_model
    return M_model, papers_model

  # ===========================
  #  Recommendation system   //
  # ===========================

  def __init__(self, model_name):
    if model_name=='TFIDF':
      df_papers = pd.read_csv(self.path_papers)
      ef = embedding_factory('TF-IDF')
      embeddings_TFIDF = ef.getEmbeddings(df_papers)
      self.papers = df_papers
      self.matrix = cosine_similarity( embeddings_TFIDF, embeddings_TFIDF )
    else:
      self.matrix, self.papers = self.get_matrix_and_papers( model_name )
    # Dictionary from doi to position in the matrix
    # Recommendations with the model
    self.position = {}
    for row2 in self.papers.index: 
      doi = self.papers['paper_id'][row2]
      self.position[doi] = row2
  
  def recommendation_by_id( self, paper_id, N = 10, print_paper=False ):
    # Recommendations with the model
    position_base = -1
    for row2 in self.papers.index: 
      if self.papers['paper_id'][row2]==paper_id: 
        position_base = row2 
    if print_paper:
      print_paper_info_by_id( self.papers, paper_id )
    return self.get_top_n(N,self.papers,self.matrix,position_base)
  
  def recommendation_by_id_threshold( self, paper_id, thre, print_paper=False ):
    # Recommendations with the model
    position_base = -1
    for row2 in self.papers.index: 
      if self.papers['paper_id'][row2]==paper_id: 
        position_base = row2 
    if print_paper:
      print_paper_info_by_id( self.papers, paper_id )
    return self.get_by_threshold(thre,self.papers,self.matrix,position_base)

  # =============
  #  Testing   //
  # =============

  def subset_test( self, paper_id, set_answer, N=10 ):
    recommendations, values = self.recommendation_by_id( paper_id, N )
    cnt = 0;
    for id in recommendations:
      if id in set_answer: cnt += 1
    return cnt
  
  def precision_subset_test_theshold( self, paper_id, set_answer, thre ):
    '''
    the is the minimum threshold to get the recommendation
    '''
    recommendations, values = self.recommendation_by_id_threshold( paper_id, thre )
    cnt = 0;
    for id in recommendations:
      if id in set_answer: cnt += 1
    
    # Calculate precision
    if len(recommendations)==0: return 0
    return cnt/len(recommendations)
  
  def average_precision_test_theshold( self, paper_id, set_answer, thre ):
    '''
    This function calculate the average precision with the formula of
    https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)
    
    * the: is the minimum threshold to get the recommendation
    '''
    recommendations, values = self.recommendation_by_id_threshold( paper_id, thre )
    cnt_relevants = 0;
    ans = 0
    for i in range(len(recommendations)):
      id = recommendations[i]
      if id in set_answer: 
        cnt_relevants += 1
        precision = cnt_relevants/(i+1)
        ans += precision/cnt_relevants
    return ans

  def get_recommendation(self, doi):
    return self.recommendation_by_id_threshold(doi,0) 
  
  def get_recommendation_V2( self, doi ):
    # Position of the article in the matrix
    pos_paper = self.position[doi]
    # Obtain the relevance of each article
    M = self.matrix 
    n = len(M[0]) if type(M[0])==list else M.shape[1]
    top_n_ids = [None]*(n-1)
    top_n_values = [None]*(n-1)
    pos_ans = 0
    for j in range(n):
      if j!=pos_paper:
        top_n_ids[pos_ans] = self.papers.iloc[j]['paper_id']
        top_n_values[pos_ans] = M[pos_paper][j] 
        pos_ans += 1
    return top_n_ids, top_n_values