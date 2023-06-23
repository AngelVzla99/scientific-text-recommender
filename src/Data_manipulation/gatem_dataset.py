# paths 
main_path = 'drive/MyDrive/Universidad/Tesis_sistema_de_recomendacion'
path_to_embeddings = main_path+'/Embeddings'
path_to_embeddings_graph = main_path+'/Embeddings_graph'
path_to_dataset = main_path+'/Dataset'
path_to_test_set = main_path+'/Conjuntos_de_prueba'
path_to_results = main_path+'/Resultados'
path_to_MLP = main_path+'/MLP_models'

import pandas as pd
import numpy as np
from sklearn.metrics import ndcg_score

class datasets_api:
  translation_path = None
  # Embedders 
  papers = None
  graphs = None
  # Paper mapper
  paper_id_number = None 
  paper_doi = None
  n_papers = None
  # Papers test
  x_test = None
  y_test = None
  # Papers train
  x_train = None
  y_train = None
  # Papers val
  x_val = None
  y_val = None

  def __init__(self, paper_embedder, graph_embedder, translation_path=""):
    self.translation_path = translation_path
    # Mapp papers
    self.make_paper_mapper()

    # Load in memory the embeddings
    papers = pd.read_csv(translation_path+path_to_embeddings+'/'+paper_embedder+'.csv')
    graphs = pd.read_csv(translation_path+path_to_embeddings_graph+'/'+graph_embedder+'.csv')
    graphs = graphs.rename( columns={'embedding':'embedding_graph'} )
    self.papers = papers
    self.graphs = graphs

    # get super embeddings ( text+graph )
    inputs = papers.merge(graphs, on='paper_id', how='inner')

    # Generate train dataset
    train = pd.read_csv(translation_path+ path_to_test_set+'/train.csv' )
    self.x_train, self.y_train = self.join_and_convert_to_numpy( inputs, train )
    # Generate test dataset
    test = pd.read_csv(translation_path+ path_to_test_set+'/test.csv' )
    self.x_test, self.y_test = self.join_and_convert_to_numpy( inputs, test )    
    # Generate train dataset
    val = pd.read_csv(translation_path+ path_to_test_set+'/val.csv' )
    self.x_val, self.y_val = self.join_and_convert_to_numpy( inputs, val )
  
  def make_paper_mapper(self):
    self.paper_id_number = {}    
    
    df = pd.read_csv(self.translation_path+ path_to_dataset+'/Paper.csv' )
    ids = df['paper_id'].to_list()

    self.paper_doi = []
    for i in range(len(ids)):
      self.paper_id_number[ids[i]] = i 
      self.paper_doi.append( ids[i] )
    
    self.n_papers = df.shape[0]

  def join_and_convert_to_numpy(self, embeddings, recommendations):
    # Make join between embeddings and recommendations
    recommendations = recommendations.rename( columns={'doi':'paper_id'} )
    all = embeddings.merge(recommendations, on='paper_id', how='inner')
    # now make list of the anwerd
    all_x = []
    all_y = []
    for index, row in all.iterrows():
      # Embedding of the paper
      embedding_text = row['embedding'][:-1]
      embedding_graph = row['embedding_graph']
      embedding = embedding_text + ';' + embedding_graph
      embedding = embedding.split(';')
      embedding = [float(x) for x in embedding]

      # Recommendations for the paper
      recommendation = np.zeros(self.n_papers)
      recommended_doi = row['recommendation'].split(';')
      recommended_point = row['points'].split(';')
      for i in range(len(recommended_doi)):
        # 
        id = self.paper_id_number[ recommended_doi[i] ]
        points = recommended_point[ i ]
        recommendation[id] = points
      # append row in the dataset
      all_x.append(embedding)
      all_y.append(recommendation)
    
    # Convert matrices to numpy
    all_x = np.array([np.array(xi) for xi in all_x])
    all_y = np.array([np.array(xi) for xi in all_y])
    
    # new embedding size = 1168
    return all_x, all_y
  
  def id_to_doi(self, id):
    return self.paper_doi[id]

  def get_representation(self, doi):
    # Get text embedding
    match_text = self.papers[ self.papers['paper_id']==doi ]
    text_embedder = match_text['embedding'].to_list()[0].split(';')[:-1]
    # Get graph embedding
    match_graph = self.graphs[ self.graphs['paper_id']==doi ]
    graph_embedder = match_graph['embedding_graph'].to_list()[0].split(';')
    # join
    embedding = text_embedder + graph_embedder
    embedding = [float(x) for x in embedding]
    return np.array(embedding)
  
  def get_dataset(self,name):
    if name=='train':
      return self.x_train, self.y_train
    elif name=='test':
      return self.x_test, self.y_test
    elif name=='val':
      return self.x_val, self.y_val
    else:
      raise "dataset name doesnt exist"
