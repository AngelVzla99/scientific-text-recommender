main_path = 'drive/MyDrive/Universidad/Tesis_sistema_de_recomendacion'
path_to_embeddings = main_path+'/Embeddings'
path_to_embeddings_graph = main_path+'/Embeddings_graph'
path_to_dataset = main_path+'/Dataset'
path_to_test_set = main_path+'/Conjuntos_de_prueba'
path_to_results = main_path+'/Resultados'
path_to_MLP = main_path+'/MLP_models'

import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch import optim
import torch.optim as optim
import pandas as pd
# Data manipulation 
import os
import sys
# Metrics
from sklearn.metrics import ndcg_score

class mf_data_api():
  doi_mapper = None
  id_mapper = None
  train_matrix = None
  test_matrix = None

  def __init__(self):
    # Load edges
    train_edges = pd.read_csv(path_to_test_set + '/train.csv')
    val_edges = pd.read_csv(path_to_test_set + '/val.csv')
    # Map dois to numbers
    self.init_mapper(path_to_dataset)
    # Create matrices
    self.create_matrices(train_edges, val_edges)

  def init_mapper(self, path_to_dataset):
    papers = pd.read_csv(path_to_dataset + '/Paper.csv')['paper_id']
    self.id_mapper = papers
    self.doi_mapper = {}
    for idx, doi in enumerate(papers):
      self.doi_mapper[doi] = idx

  def from_doi_to_id(self, doi):
    return self.doi_mapper[doi]
  
  def from_id_to_doi(self, doi):
    return self.id_mapper[doi]

  def create_matrices(self, train_edges, val_edges):
    # Init matricse
    n_papers = len(self.doi_mapper)
    train_matrix = torch.zeros((n_papers,n_papers))
    # val_matrix = torch.zeros((n_papers,n_papers))
    # Create train matrix
    for index, row in train_edges.iterrows():
      fr = self.from_doi_to_id( row['doi'] )
      recommendation = row['recommendation'].split(';')
      recommendation = [ self.from_doi_to_id(x) for x in recommendation ]
      for rec_id in recommendation:
        train_matrix[fr][rec_id] = 1
        # val_matrix[fr][rec_id] = 1
    # Create val matrix
    # for index, row in val_edges.iterrows():
    #   fr = self.from_doi_to_id( row['doi'] )
    #   recommendation = row['recommendation'].split(';')
    #   recommendation = [ self.from_doi_to_id(x) for x in recommendation ]
    #   for rec_id in recommendation:
    #     val_matrix[fr][rec_id] = 1
    self.train_matrix = train_matrix
    # self.val_matrix = val_matrix