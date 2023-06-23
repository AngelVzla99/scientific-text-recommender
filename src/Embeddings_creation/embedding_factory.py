# Dependencies
# !pip install -q -U "tensorflow-text==2.8.*"
# !pip install -q tf-models-official==2.7.0
# !pip install --upgrade transformers==4.2
# !pip install transformers datasets accelerate nvidia-ml-py3 # nvidia
# !pip install --quiet bitsandbytes
# !pip install --quiet --upgrade transformers # Install latest version of transformers
# !pip install --quiet --upgrade accelerate
# !pip install --quiet sentencepiece
# !pip install transformers bitsandbytes accelerate

from public_models import *
# Lematization
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
nltk.download('omw-1.4')
# Data manipulation 
import os
import sys
import csv
import pandas as pd
import re
import math
from google.colab import output # for line cleaning
import torch
import math

# Paths used 
main_path = 'drive/MyDrive/Universidad/Tesis_sistema_de_recomendacion'
path_test_set = main_path+'/Conjuntos_de_prueba'
path_dataset = main_path+'/Dataset'

class embedding_factory():
  embedding_function = None
  dictionary_input = False

  # ===========================
  #           Models         //
  # ===========================

  def __init__(self, model_name):
    if model_name == "TF-IDF":
      self.embedding_function = self.TF_IDF
    elif model_name == "doc2vec":
      self.embedding_function = self.doc2vec
    elif model_name == "glove":
      self.embedding_function = self.glove
    elif model_name == "bert":
      self.embedding_function = self.bert
    elif model_name == "bert_L1_LN":
      self.embedding_function = self.bert_L1_LN
    elif model_name == "specter":
      self.embedding_function = self.specter
      self.dictionary_input = True
    elif model_name == 'specter_L1_LN':
      self.embedding_function = self.specter_L1_LN
      self.dictionary_input = True
    elif model_name == "scincl":
      self.embedding_function = self.scincl
      self.dictionary_input = True
    elif model_name == "scincl_L1_LN":
      self.embedding_function = self.scincl_L1_LN
      self.dictionary_input = True
    elif model_name == "bert_large":
      self.embedding_function = self.bert_large
      
  def TF_IDF( self, list_of_texts ):
    new_list = [ x.lower() for x in list_of_texts ]
    #calling the TfidfVectorizer
    vectorize = TfidfVectorizer()
    #fitting the model and passing our sentences right away:
    response = vectorize.fit_transform(new_list)    
    # cosine_values = cosine_similarity(response[0], response)[0]
    # for val in cosine_values:
    #   print(val)
    return response
  
  def doc2vec(self, list_of_texts):
    '''
    documentation: https://radimrehurek.com/gensim/models/doc2vec.html
    '''
    tokenized_doc = []
    for d in list_of_texts:
      tokenized_doc.append(word_tokenize(d.lower()))
    # Convert tokenized document into gensim formated tagged data
    tagged_data = [TaggedDocument(d, [i]) for i, d in enumerate(tokenized_doc)]
    ## Train doc2vec model ( distributed bag of words (PV-DBOW) dm = 0 default )
    model = Doc2Vec(tagged_data, vector_size=128, window=2, min_count=1, workers=4, epochs = 1000)
    # find most similar doc 
    test_doc = word_tokenize(list_of_texts[0].lower())
    #ans = model.docvecs.most_similar(positive=[model.infer_vector(test_doc)],topn=10)
    #new_ans = [ 0 for x in list_of_texts ]
    #for (p,val) in ans: new_ans[p] = val
    #for val in ans: print(val)
    ans = [ model.infer_vector(doc.lower()) for doc in list_of_texts ]
    return ans

  def glove(self, list_of_texts):    
    vectors = []
    for text in list_of_texts:
      vec = np.zeros(25) # len 25 of the glove embeddings
      for word in text.split(' '):
        sample_glove_embedding = glove_model[word.lower()]
        np_array = np.asarray( sample_glove_embedding )
        vec += np_array
      vectors.append(vec)

    for i in range(len(vectors)):
      vec1 = [vectors[0]]
      vec2 = [vectors[i]]
      print(cosine_similarity(vec1,vec2)[0][0])
    return [[1,2,3],[1,2,3]]

  def bert(self, list_of_texts):
    '''
    Documentation:
    https://www.tensorflow.org/text/tutorials/classify_text_with_bert
    ->
    https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4
    -> (fine tuning in glue problems)
    https://colab.research.google.com/github/tensorflow/text/blob/master/docs/tutorials/bert_glue.ipynb#scrollTo=qXU6bQWmNfhp
    '''

    preprocess_text = bert_preprocessor(list_of_texts)
    # embeding de [CLS] [tk1] [tk2] ...
    embeddings = bert_encoder(preprocess_text)['pooled_output']
    ans = [ emb.numpy() for emb in embeddings ]
    del embeddings
    del preprocess_text
    return ans

  def bert_L1_LN(self, list_of_texts):
    '''
    Documentation:
    https://www.tensorflow.org/text/tutorials/classify_text_with_bert    
    '''

    preprocess_text = bert_preprocessor(list_of_texts)
    # embeding de [CLS] [tk1] [tk2] ...
    embeddings = bert_encoder(preprocess_text)['encoder_outputs']

    L1 = embeddings[0].numpy()
    LN = embeddings[len(embeddings)-1].numpy()
    
    ans = []
    for text in range(len(list_of_texts)):
      # embedding[i] = avg of embedings in the word i
      embeddingL1 = np.average(L1[text], axis=0)
      embeddingLN = np.average(LN[text], axis=0)            
      # embedding = avg between each layer
      embedding = (embeddingL1+embeddingLN)/2
      ans.append(embedding)      
      del embedding
      del embeddingL1
      del embeddingLN
    
    del L1
    del LN
    
    return ans

  def specter(self,papers): 
    # concatenate title and abstract
    # [CLS] tx1 [SEP] tx2
    title_abs = [d['title'] + tokenizer.sep_token + (d.get('abstract') or '') for d in papers]
    # preprocess the input
    inputs = tokenizer(title_abs, padding=True, truncation=True, return_tensors="pt", max_length=512).to("cuda:0")
    result = model(**inputs)
    # take the first token in the batch as the embedding
    embeddings = result.last_hidden_state[:, 0, :]
    # compare embeddings
    # for i in range(len(embeddings)):
    #   print(cosine_similarity([embeddings[0].detach().numpy()],[embeddings[i].detach().numpy()])[0][0])
    return [ emb.cpu().detach().numpy() for emb in embeddings ]

  def specter_L1_LN(self,papers):    
    '''
    How to get each layer: 
    documentation : https://huggingface.co/docs/transformers/v4.24.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput.last_hidden_state
    '''
    if isinstance(papers[0],str):
      papers = [ {'title':x} for x in papers ]

    # concatenate title and abstract
    # [CLS] tx1 [SEP] tx2
    title_abs = [d['title'] + tokenizer.sep_token + (d.get('abstract') or '') for d in papers]
    # preprocess the input
    inputs = tokenizer(title_abs, padding=True, truncation=True, return_tensors="pt", max_length=512).to("cuda:0")
    result = model(**inputs, output_hidden_states=True)
    # take the first token in the batch as the embedding
    embeddings = result.hidden_states

    L1 = embeddings[1]
    L1 = tf.convert_to_tensor(L1.cpu().detach().numpy(),dtype=tf.float64)
    LN = embeddings[12] # because there are 12 layers
    LN = tf.convert_to_tensor(LN.cpu().detach().numpy(),dtype=tf.float64)
    ans = []
    for text in range(len(papers)):      
      embeddingL1 = np.average(L1[text], axis=0)
      embeddingLN = np.average(LN[text], axis=0)            
      # embedding = avg between each layer            
      embedding = (embeddingL1+embeddingLN)/2
      ans.append(embedding)
      del embedding
      del embeddingL1
      del embeddingLN
    del embeddings
    return ans

  def scincl(self,papers):
    if isinstance(papers[0],str):
      papers = [ {'title':x} for x in papers ]
    
    # concatenate title and abstract
    # [CLS] tx1 [SEP] tx2
    title_abs = [d['title'] + tokenizer_scincl.sep_token + (d.get('abstract') or '') for d in papers]
    # preprocess the input
    inputs = tokenizer_scincl(title_abs, padding=True, truncation=True, return_tensors="pt", max_length=512).to("cuda:0")
    result = model_scincl(**inputs)
    # take the first token in the batch as the embedding
    embeddings = result.last_hidden_state[:, 0, :]
    # tensors to numpy arrays
    embeddings = [ emb.detach().cpu().numpy() for emb in embeddings ]
    return embeddings

  def scincl_L1_LN(self,papers):    
    # concatenate title and abstract
    # [CLS] tx1 [SEP] tx2
    title_abs = [d.get('title') + tokenizer_scincl.sep_token + (d.get('abstract') or '') for d in papers]
    # preprocess the input
    inputs = tokenizer_scincl(title_abs, padding=True, truncation=True, return_tensors="pt", max_length=512).to("cuda:0")
    result = model_scincl(**inputs, output_hidden_states=True)
    # take the first token in the batch as the embedding
    embeddings = result.hidden_states

    L1 = embeddings[1]
    L1 = tf.convert_to_tensor(L1.cpu().detach().numpy(),dtype=tf.float64)
    LN = embeddings[12] # because there are 12 layers
    LN = tf.convert_to_tensor(LN.cpu().detach().numpy(),dtype=tf.float64)
    ans = []
    for text in range(len(papers)):      
      # embeddingL1 = self.avg_matrix(L1[text])
      # embeddingLN = self.avg_matrix(LN[text])
      embeddingL1 = np.average(L1[text], axis=0)
      embeddingLN = np.average(LN[text], axis=0)            
      # embedding = avg between each layer            
      embedding = (embeddingL1+embeddingLN)/2
      ans.append(embedding)
      del embedding
      del embeddingL1
      del embeddingLN
    del embeddings
    return ans

  def bert_large(self,papers):    
    '''
    https://huggingface.co/bert-large-cased
    512 input tokens 
    1024 output tokens
    '''
    
    # preprocess the input
    encoded_input = tokenizer_bert_large(papers, padding=True, truncation=True, return_tensors="pt", max_length=512).to("cuda:0")
    result = model_bert_large(**encoded_input, output_hidden_states=True)
    # take the first token in the batch as the embedding
    embeddings = result.hidden_states

    L1 = embeddings[1]
    L1 = tf.convert_to_tensor(L1.cpu().detach().numpy(),dtype=tf.float64)
    LN = embeddings[24] # because there are 24 layers
    LN = tf.convert_to_tensor(LN.cpu().detach().numpy(),dtype=tf.float64)
    ans = []
    for text in range(len(papers)):      
      # embeddingL1 = self.avg_matrix(L1[text])
      # embeddingLN = self.avg_matrix(LN[text])
      embeddingL1 = np.average(L1[text], axis=0)
      embeddingLN = np.average(LN[text], axis=0)            
      # embedding = avg between each layer            
      embedding = (embeddingL1+embeddingLN)/2
      ans.append(embedding)
      del embedding
      del embeddingL1
      del embeddingLN
    del embeddings
    return ans

  def text_preprocessing(self, paper):
    '''
    different lematizers in python:
      https://www.geeksforgeeks.org/python-lemmatization-approaches-with-examples/
    '''
    # Create WordNetLemmatizer object
    wnl = WordNetLemmatizer()  
    # Converting String into tokens 
    list2 = nltk.word_tokenize(paper)      
    # Lematize and join the tokens
    lemmatized_string = ' '.join([wnl.lemmatize(words) for words in list2])
    # delete stop words *** (?) 

    return lemmatized_string

  def getEmbeddings( self, data_frame_papers, paper_representation="ALL" ):
    # Convert our dataset to a good format
    papers = []
    papers_dictionary = []
    for index, row in data_frame_papers.iterrows():    
      # Row to strings 
      if paper_representation=="ALL":
        ans = str(row['title']) + " " + str(row['category']) + " " + str(row['keywords']) + " " + str(row['abstract'])
        ans_dictionary = { 'title':str(row['title']) + " " + str(row['category']) + " " + str(row['keywords']), 'abstract' : str(row['abstract']) }
      elif paper_representation=="title&abstract":
        ans = str(row['title']) + " " + str(row['abstract'])
        ans_dictionary = { 'title':str(row['title']), 'abstract' : str(row['abstract']) }      
      elif paper_representation=="title":
        ans = str(row['title'])
        ans_dictionary = { 'title':str(row['title']), 'abstract' : '' }      
      # Preprocessing
      ans = self.text_preprocessing(ans)      
      ans_dictionary['title'] = self.text_preprocessing(ans_dictionary['title'])
      ans_dictionary['abstract'] = self.text_preprocessing(ans_dictionary['abstract'])
      # Add to samples
      papers.append(ans)
      papers_dictionary.append(ans_dictionary)    

    if self.dictionary_input:
      return self.embedding_function(papers_dictionary)
    return self.embedding_function(papers)