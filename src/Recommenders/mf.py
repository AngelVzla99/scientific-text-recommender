# paths
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
import mf_data_api

prox_plus = nn.Threshold(0,0) # to make all output postive 

torch.manual_seed(7)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

class MF_MODEL(nn.Module): 
  def __init__(self, u, v, d):
    super(MF_MODEL, self).__init__()
    self.U = nn.Parameter(torch.rand(u, d)/10000, requires_grad=True)
    # self.V = nn.Parameter(torch.rand(d, v)/100, requires_grad=True)

    self.act_func = nn.Tanh()

  def forward(self):
    # score = torch.matmul(self.U, self.V)
    score = torch.matmul(self.U, torch.transpose(self.U, 0, 1))
    return score

def train( Xtorch, d = 50, n_epoch = 1000, lr=0.00001, verbose=True, print_plots=True, save_model=True ):
  model_name = "MF_d:"+str(d)+"_nepochs:"+str(n_epoch)
  # Set dimension parpemeter 
  u = Xtorch.shape[0]
  v = Xtorch.shape[1]
  # Define model, loss function and optimizer (SGD)
  model = MF_MODEL(u,v,d)
  loss_fn = nn.MSELoss(reduction='sum')
  optimizer = optim.SGD(model.parameters(), lr=lr, momentum = 0.9)
  # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.001)
  # Training process
  model_loss=[]
  for epoch in range(n_epoch):
    # Model prediction
    X_= model()
    # Back-propagation
    loss = loss_fn(X_, Xtorch)
    model.zero_grad() # need to clear the old gradients
    loss.backward()
    optimizer.step()
    # Decrease of the lr
    # scheduler.step()
    # Model save
    if (epoch+1)%50==0 and (epoch+1)>0 and save_model:
      torch.save(model, path_to_MLP+'/'+model_name+'.pth')
      if verbose: print("Saved "+model_name)
    model_loss.append(loss.detach().numpy())
    if verbose: print("Epoch "+str(epoch) +   "\t| training error: " +str(model_loss[-1]))
  # plots of the training process
  if print_plots:
    plt.plot(model_loss)
    plt.ylabel('loss over time')
    plt.xlabel('iteration times')
    plt.show()
  print("The training for the model "+ model_name + " has ended")
  print('Final loss: '+str(model_loss[-1]))
  return model_loss[-1]

class matrix_factorization:
  model = None
  api = None
  matrix = None

  def __init__(self, d, n_epoch):
    # api to get data
    self.api = mf_data_api()
    self.model_name = 'test'
    
    model_name = "MF_d:"+str(d)+"_nepochs:"+str(n_epoch)
    self.model = torch.load(path_to_MLP+'/'+model_name+".pth").to('cpu')
    # self.model = model 
    self.model.eval()
    self.matrix = self.model()
  
  def get_recommendation_V2( self, doi ):
    # map doi to position in the matrix
    row_index = self.api.from_doi_to_id( doi )
    with torch.no_grad():
      # Get relevance for each article
      row = self.matrix.index_select(0, torch.tensor([row_index]) )
      relevance = row.cpu().numpy()[0]      
      # Map number in the matrix to doi
      dois = [ self.api.from_id_to_doi(x) for x in range(len(relevance)) ]
      return dois, relevance
    return None