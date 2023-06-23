# paths
main_path = 'drive/MyDrive/Universidad/Tesis_sistema_de_recomendacion'
path_to_embeddings = main_path+'/Embeddings'
path_to_dataset = main_path+'/Dataset'
path_to_test_set = main_path+'/Conjuntos_de_prueba'
path_to_results = main_path+'/Resultados'
path_to_MLP = main_path+'/MLP_models'
path_to_embeddings_graph = main_path+'/Embeddings_graph'

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

class main_tester:
  # Test cases
  tests = []
  # Recomendation system
  system = None
  system_threshold = None
  # Model specifications
  recommender_name = None
  recommender_prefix = None

  def __init__(self, model_prefix, model_suff='', model_suff_2 = '', model_suff_3 = '', model_suff_4 = ''):
    # Load system
    if model_prefix == "MS":
      self.system = microsoft_recommender( model_suff, model_suff_2, model_suff_3, model_suff_4 )
    elif model_prefix == "EM":
      self.system = emb_recommender( model_suff )
    elif model_prefix == "GATEM":
      self.system = gatem( model_suff, model_suff_2, model_suff_3 )
    elif model_prefix=="CF":
      self.system = collaborative_filtering_recommender( model_suff )
    elif model_prefix == "MF":
      self.system = matrix_factorization( model_suff, model_suff_2 )
    else:
      raise ValueError('First parameter model dont exist.')
    self.recommender_name = model_prefix+"_"+str(model_suff)+"_"+str(model_suff_2)+"_"+str(model_suff_3)+"_"+str(model_suff_4)
    self.recommender_prefix = model_prefix

    # Load threshold    
    # with open(path_to_testing+'/thresholds.json') as json_data:
    #   th_dic = json.load(json_data)
    #   self.system_threshold = th_dic[model_name]
    # Load test from dataset ( mod7 )

    # if len(self.tests)==0:
    #   full_test = pd.read_csv(path_to_test_set+'/full_test.csv').head(20)
    #   for index, row in full_test.iterrows():
    #     articles = row['recommendation'].split(';')
    #     # Take data from df
    #     id = row['doi']
    #     dois = row['recommendation'].split(';')
    #     points = row['points'].split(';')
    #     # Convert data to dictionary
    #     articles = {}
    #     for i in range(len(dois)):
    #       articles[dois[i]] = points[i]
    #     self.tests.append({ 'doi':id, 'ans':articles })
  
  # ============= //
  #    Metrics    //
  # ============= //

  def nDCG(self, pred, y_true=None): 
    pred = np.asarray([pred])
    if y_true==None:
      y = -np.sort(-pred[0])
      y = np.asarray([y])
    else: 
      y = np.asarray([y_true])
    return ndcg_score(y, pred)

  def kprecision(self, relevance, true_relevance,k=10):
    result = [None]*len(relevance)
    for i in range(len(relevance)):
      result[i] = [ relevance[i], true_relevance[i] ]
    result.sort(); result.reverse()

    ans_max = 0
    ans = 0
    for i in range(len(result)):
      val = result[i][1]
      if val>0: ans_max+=1
      if val>0 and i<k: ans+=1
    return ans/ans_max

  def aveP(self, result):
    result = [ 1 if x>0 else 0 for x in result ]
    # Sum of relevant documents
    sum_relevant = sum(result)
    
    # Iterate over results
    relevant_in_top_k = 0
    cu_relevant = 0
    ans = 0
    for k in range(len(result)):
      # current number of relevant doc
      if relevant_in_top_k<sum_relevant:
        relevant_in_top_k += 1

      # Calculate precision
      cu_relevant += result[k]
      precision = cu_relevant/relevant_in_top_k

      # Sum if precision 
      if result[k]>0:
        ans += precision/relevant_in_top_k

    return ans

  # ====================== //
  #    Useful functions    //
  # ====================== //

  def nonseed(self):
    print("=== test nonseed ===")

    # Load queries 
    queries_df = pd.read_csv(path_to_test_set+'/test_set_nonseeded.csv')
    queries = queries_df['text'].to_list()
    
    ans_all_test = {}
    index = 0
    for query in queries:
      # Query using threshold
      recommendation, relevance = self.system.search( query, False, -1, self.system_threshold )
      ans_all_test["Query_"+str(index)] = len(recommendation)
      print( len(recommendation) )
      index += 1
    return ans_all_test
  
  def best_threshold_by_contains(self):
    '''
    This test will find the threshold that will give us all the resuls
    '''
    print("=== test best threshold by contains ===")
    
    # Iterate over all test cases to get the minimum threshold for each one
    thresholds_by_test = np.zeros( len(self.tests) )
    ans_all_test = {}
    index_tex = 0
    for i_test in range(len(self.tests)):
      # Binary search to get the minimum threshold that give us all good papers 
      test = self.tests[i_test]
      best_value = 100 # Guessing that 100 is greater that all the relevance
      number_of_papers = self.system.df_user_info.shape[0]
      results, relevance = self.system.search( test['query'] )
      for index in range(len(results)):
        id = results[index]
        value = relevance[index]
        if id in test['ans']:
          best_value = value
      
      # Save the value to be retuned
      ans_all_test["Query_"+str(index_tex)] = best_value
      index_tex += 1

      # Here we have the minimum threshold to have all the good papers
      thresholds_by_test[i_test] = best_value
      print(best_value)
      # print(str(best_value).replace('.',','))
    
    # We have the minimums of each tetst in `thresholds_by_test`
    #print( np.amin(thresholds_by_test) )    
    return ans_all_test   

  def metrics(self, metric_name = 'nDCG'):
    print("=== test ranking "+metric_name+" ===")
    ans_all_test = {}
    index = 0
    values = np.zeros( len(self.tests) )
    for idx, test in enumerate(self.tests):
      # sum 1/log(i) if i-th id is good 
      ids, full_relevance = self.system.get_recommendation_V2( test['doi'] )
      
      # uncomment to know the recommendations
      # print("\n\ndado "+str(test['doi']))
      # print("quiero "+str(test['ans']))
      # print("Lo que da el recomendador es: ")
      true_relevance = []
      relevance = []
      for i in range(len(ids)):
        if not ids[i] in test['ignore']:
          # if i < 10: print(ids[i])
          relevance.append(full_relevance[i])
          if ids[i] in test['ans']:
            true_relevance.append(test['ans'][ids[i]])
          else:
            true_relevance.append(0)    
        
      if metric_name=="nDCG":
        metric = self.nDCG( relevance, true_relevance )
      elif metric_name=="kprecision":
        metric = self.kprecision( relevance, true_relevance )
      # ans_all_test["Query_"+str(idx)] = round(metric,4)
      values[idx] = round(metric,4)
    
    print("La metrica "+metric_name+" nos dio "+str(values.mean()))

    return values.mean() # ans_all_test 

  def time_query(self):
    print("=== test of time ===")
    ans_all_test = {}
    index = 0
    values = []
    for test in self.tests:
      # sum 1/log(i) if i-th id is good 
      start = time.time()
      ids, relevace = self.system.search( test['query'] )      
      end = time.time()
      t = end - start
      
      ans_all_test["Query_"+str(index)] = round(t)
      values.append(t)
      index += 1
    return ans_all_test 
  
  def time_query_embedding_creation(self):
    print("=== test of time (embedding creating) ===")
    ans_all_test = {}
    index = 0
    values = []
    for test in self.tests:
      start = time.time()
      embeddings = self.system.factory.get_embedding_from_text(test['query'])
      end = time.time()
      t = end - start
      
      ans_all_test["Query_"+str(index)] = t
      print(t)
      values.append(t)
      index += 1
    return ans_all_test 

  def run_all_datasets(self, action=None, metric_name = "nDCG"):
    datasets = []
    datasets.append( 'seed_test' )
    datasets.append( 'new_seed_test' )
    datasets.append( 'test_author' )
    datasets.append( 'test_citations' )
    datasets.append( '312_test' )
    datasets.append( 'citations_path_test' ) 

    # Helper function to done the same shape to all tha data
    def from_dic_to_df( ans ):
      list_df = []
      for val in ans.keys():
        # Convert to strigns
        representation = ''
        points = ''
        for doi in ans[val]: 
          representation += doi[0] + ";"
          points += str(doi[1]) + ";"
        representation = representation[:-1]
        points = points[:-1]
        list_df.append({'doi':val, 'recommendation':representation, 'points':points})

      return pd.DataFrame(list_df) 

    def from_edge_to_list( all_data ):
      # Transform edges to adyacent list 
      ans = {}
      for index, row in all_data.iterrows():
        doi1 = row['from']
        doi2 = row['to']
        point = row['points']
        if not( doi1 in ans ): ans[doi1] = []
        ans[doi1].append([doi2,point])
      return from_dic_to_df(ans)

    # Dictionary, dic[a][b] = true if given a, youi should give me b
    full_recommendation = {}
    full_dataset = pd.read_csv(path_to_test_set+'/full_test.csv')
    for index, row in full_dataset.iterrows():
      # Take data from df
      id = row['doi']
      full_recommendation[id] = row['recommendation'].split(';')

    results = { 'model':self.recommender_name }
    for dataset in datasets:
      print("\nUsando dataset "+str(dataset))
      # Load dataset in memory
      self.tests = []
      test_df = pd.read_csv(path_to_test_set+'/'+dataset+'.csv')

      if dataset!='full_test' and dataset!='test':
        # Convert to: given 'a' give me ['b','c','d']
        test_df = from_edge_to_list(test_df)


      for index, row in test_df.iterrows():
        articles = {}
        # Take data from df
        id = row['doi']
        dois = row['recommendation'].split(';')
        points = row['points'].split(';')
        # Convert data to dictionary
        for i in range(len(dois)):
          articles[dois[i]] = float(points[i])
        # Select papers to be ignored in the test
        dic_sub_test = {}
        for doi in dois: dic_sub_test[doi] = True
        ignore = {}
        if id in full_recommendation:
          for doi in full_recommendation[id]:
            if not doi in dic_sub_test:
              ignore[doi] = True
        
        if dataset == 'full_test': ignore = {}
        self.tests.append({ 'doi':id, 'ans':articles, 'ignore': ignore })
      
      print("Cantidad de articulos "+str(len(self.tests)))
      # Run a test in this dataset
      score = self.metrics(metric_name)
      results[dataset] = score
    results_df = pd.DataFrame([results])

    if action!=None:
      path_to_save_results = path_to_results+'/'+self.recommender_prefix+'_'+'nDCG'+'.csv'
      if action=="create" or (not os.path.isfile(path_to_save_results)):
        print("Saving test (creation)")
        results_df.to_csv(path_to_save_results, index=False)
      elif action=="update":
        print("Saving dataset (update)")
        old_df = pd.read_csv(path_to_save_results)
        # Append to the dataframe and save
        results_df = pd.concat([old_df,results_df])
        results_df.to_csv(path_to_save_results, index=False)
      else:
        print(str(action)+" is not a valida action")
  
  def run_all(self):
    val_contains =  self.contains()
    print("\n")
    val_best_th = self.best_threshold_by_contains()
    print("\n")
    val_ranking = self.ranking_precision()
    print("\n")
    val_nonseed = self.nonseed()
    print("\n")
    val_time =  self.time_query()
    print("\n")
    return [val_contains, val_best_th,val_ranking,val_nonseed,val_time]