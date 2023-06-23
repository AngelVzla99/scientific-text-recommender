# Paths used 
main_path = 'drive/MyDrive/Universidad/Tesis_sistema_de_recomendacion'
path_test_set = main_path+'/Conjuntos_de_prueba'
path_dataset = main_path+'/Dataset'
path_embeddings = main_path+'/Embeddings'

import pandas as pd
import numpy as np
import math
import os
from IPython import display as output

class embedding_manager():
  path = path_embeddings
  path_papers = path_dataset+'/Paper.csv'
  
  def append_rows(self, papers_ids, embeddings, file_name):
    # Iterate over the dataframe  
    publications = []
    for index in range(len(papers_ids)):
      id = papers_ids[index]
      emb = ''
      for x in embeddings[index]: emb += str(x) + ";"
      publications.append( { 'paper_id':papers_ids[index], 'embedding': emb } )
    
    # take the saved embeddings
    path_current_embeddings = self.path+"/"+file_name+".csv"
    col_names = ['paper_id','embedding']
    old_df  = pd.DataFrame(columns = col_names)
    if os.path.isfile(path_current_embeddings):
      old_df = pd.read_csv(path_current_embeddings, error_bad_lines=False)
    
    # Append to the dataframe and save
    new_df = pd.DataFrame( publications )
    new_df = old_df.append(new_df)
    new_df.to_csv(self.path+"/"+file_name+".csv", index=False)
    del new_df

  def save(self, papers_df, file_name, model_name, paper_representation="ALL"):
    '''
    where paper_representation = title&abstract | ALL | title
    '''
    # Get the embeddings for each paper
    start = 0
    batch = 1 # 40 with collab pro, and 5 with normal collab
    batch_save = 300
    saved_embeddings = []
    saved_papers_ids = []

    # If there is data processesed before
    path_current_embeddings = self.path+"/"+file_name+".csv"
    if os.path.isfile(path_current_embeddings):
      prev_df = pd.read_csv(path_current_embeddings, error_bad_lines=False)
      start = prev_df.shape[0]+1
      del prev_df

    ef = embedding_factory(model_name)
    for it in range(math.ceil((papers_df.shape[0]-start)/batch)):
      # load bar
      output.clear()
      print( (start/batch)/(math.ceil(papers_df.shape[0]/batch)) )

      # subset of papers for this batch
      mi = (start+batch) if (start+batch) < papers_df.shape[0] else papers_df.shape[0]
      r = [x for x in range(start,mi)]
      sub_df = papers_df.take(r)

      # Get the embeddings of all the sub_df as numpy array
      embeddings = ef.getEmbeddings(sub_df,paper_representation)
      embeddings_without_gradient = embeddings

      # Make accumulation of the embeddings
      saved_embeddings = saved_embeddings + embeddings_without_gradient
      saved_papers_ids = saved_papers_ids + sub_df['paper_id'].tolist() 

      # If we reach the batch size => save it in a file
      if (batch_save <= len(saved_embeddings)) or (mi==papers_df.shape[0]):
        self.append_rows(saved_papers_ids, saved_embeddings, file_name)
        del saved_embeddings
        del saved_papers_ids
        saved_embeddings = []
        saved_papers_ids = []
        print("Backup")

      start += batch
      del sub_df
      del embeddings
    
    # load bar
    #output.clear()
    print( (start/batch)/(math.ceil(papers_df.shape[0]/batch)) )

  def load(self, file_name):
    return pd.read_csv(self.path+'/'+file_name+".csv", error_bad_lines=False) 

  def load_and_join(self, file_name):
    embeddings = self.load(file_name)
    df_papers_join = pd.read_csv(self.path_papers, error_bad_lines=False)
    return pd.merge(embeddings.dropna(), df_papers_join.dropna(), left_on=['paper_id'], right_on=['paper_id'], how='inner')    

  def check_contains(self,file_name, model_name, paper_representation):
    # papers
    df_papers = pd.read_csv(self.path_papers, error_bad_lines=False)    
    # embeddings 
    df = self.load_and_join(file_name)
      
    # convert ids to list
    v1 = df['paper_id'].tolist() 
    v2 = df_papers['paper_id'].tolist() 
    # print the ones in v2 that aren't in v1
    pos = 0
    # Accumulate the missing papers
    r = []
    for val2 in v2:
      is_here = False
      for val1 in v1:
        if val1==val2: is_here = True
      if not is_here:
        row = df_papers.iloc[pos]
        r.append(pos)
        print(str(val2)+ " " +str(pos)) 
      pos += 1

    if len(r)>0:
      print("There are papers  in df_papers that aren't in the embedding file")
      # Add the resting rows
      embeddings = []
      paper_ids = []
      ef = embedding_factory(model_name)
      for row in r:
        sub_df = df_papers.take([row])
        sub_paper_ids = sub_df['paper_id'].tolist() 
        sub_embeddings = ef.getEmbeddings(sub_df,paper_representation)
        # Save and clean memory
        embeddings.append(sub_embeddings[0])
        paper_ids.append(sub_paper_ids[0])
        del sub_df
        del sub_paper_ids
        del sub_embeddings

      # Tensor to numpy arrays
      embeddings_without_gradient = embeddings
      self.append_rows(paper_ids, embeddings_without_gradient, file_name)
      print("Rows added "+str(len(r)))
    print("Cantidad correcta de embeddings !!")

  def get_numpy_embedding(self, list_of_strings):
    ans = []
    for s in list_of_strings:
      np_array_paper = None
      if ';' in s: 
        np_array_paper = s.split(';')[:-1]
        np_array_paper = [ float(x) for x in np_array_paper ]
        np_array_paper = np.asarray(np_array_paper)
      else : 
        np_array_paper = np.fromstring(s[1:-1], dtype=float, sep=' ')
      ans.append(np_array_paper)
    return ans