# Paths
main_path = 'drive/MyDrive/Universidad/Tesis_sistema_de_recomendacion'
path_test_set = main_path+'/Conjuntos_de_prueba'
path_dataset = main_path+'/Dataset'

import os
import pandas as pd
import numpy as np
import re

def open_csv( path, ans ):
  # There are articles in english and spanish
  dataFrame = pd.read_csv(path, on_bad_lines=False)
  # We make a projection of the dataset
  dataFrame = dataFrame[ ['Author(s) ID','DOI', 'Title', 'Abstract', 'Author Keywords', 'Source title'] ]
  # Concat the answer
  ans = pd.concat([ans, dataFrame],ignore_index=True)
  return ans

# Apply a function to all the files in a folder and accumulate the answers
def apply_func( source_folder, func, ans ):
  dirfiles = os.listdir(source_folder)
  fullpaths = map(lambda name: os.path.join(source_folder, name), dirfiles)
  file_format = ".csv"

  # Apply the function to only the files with the right format and make 
  # it recursice over folders
  for file in fullpaths:
      if os.path.isdir(file):
        ans = apply_func(file,func,ans)
      if os.path.isfile(file) and file.endswith( file_format ):         
        ans = func(file, ans)
  return ans

def read_raw_data():
  ans = pd.DataFrame()
  dirname = 'drive/MyDrive/Universidad/Tesis/RAW_DATA'
  ans = apply_func( dirname, open_csv, ans )

  # Delete null rows
  ans = ans.dropna(axis=0, how='any', subset=None, inplace=False)
  # Delete duplicated rows
  ans = ans.drop_duplicates()
  # Eliminate data with bad format
  df2 = []
  for index, row in ans.iterrows():
    ids = row['Author(s) ID']
    # delete spanish in the title
    my_regex_spanish = "\[.*\]"
    row['Title'] = re.sub(my_regex_spanish, "", row['Title'])
    # delete the rigth reserved
    my_regex_rights_reserved = "\s©\s.*"
    row['Abstract'] = re.sub(my_regex_rights_reserved, "", row['Abstract'])
    if ids[-1]==';' and len(row['Title'])>0 and len(row['Abstract'])>0:
      df2.append(row)
  ans = pd.DataFrame(df2)
  # The abstract and the DOI have to be unique
  print("Antes => "+str(ans.shape[0])) # 99188
  ans = ans.drop_duplicates(subset=['Abstract'])
  ans = ans.drop_duplicates(subset=['Title','Author Keywords'])
  ans = ans.drop_duplicates(subset=['DOI'])

  # Save the dataFrame
  #ans.to_csv('drive/MyDrive/Universidad/Tesis/DATASETS/PAPERS.csv', index=False)
  print( str(ans.shape[0]) + " " + str(ans.shape[1]) )

  return ans

# This join will add to the dataset the category of each paper
def join_with_category( publications ):
  # Get all the publishers with their name and category
  dirname = 'drive/MyDrive/Universidad/Tesis/RAW_DATA'
  columns_publisher = ["Source Title (Medline-sourced journals are indicated in Green)",'All Science Journal Classification Codes (ASJC)']
  pub = pd.read_excel(dirname+'/extlistAugust2022-2.xlsx', sheet_name='Scopus Sources May 2022', header=0, names=None, index_col=None, usecols=columns_publisher) 
  print("Total de Publishers = "+str(pub.shape[0]))
  pub = pub.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)  
  print("Total de Publishers sin NULL = "+str(pub.shape[0]))
  pub = pub.drop_duplicates()
  print("Total de Publishers sin NULL  y sin DUPLICADOS = "+str(pub.shape[0]))
  pub = pub.drop_duplicates(subset=[columns_publisher[0]])
  print("Total de Publishers sin NULL  y sin DUPLICADOS, con primera columna unica = "+str(pub.shape[0]))
  # Get all the scopus categories
  dirname_target = 'drive/MyDrive/Universidad/Tesis/DATASETS'
  categories = pd.read_csv(dirname_target+'/categories.csv', error_bad_lines=False)
  # Make the joins between the publications and publishers to get the categories
  df_join = pd.merge(categories.dropna(), pub.dropna(), left_on=['id'], right_on=[columns_publisher[1]], how='inner')
  df_join2 = pd.merge(df_join.dropna(), publications.dropna(), left_on=[columns_publisher[0]], right_on=['Source title'], how='inner')
  dataFrame = df_join2[ ['Author(s) ID','DOI', 'Title', 'Abstract', 'Author Keywords', 'Category'] ]
  dataFrame.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
  dataFrame = dataFrame.drop_duplicates()
  print( str(publications.shape[0]) + " " + str(publications.shape[1]) )
  print( str(dataFrame.shape[0]) + " " + str(dataFrame.shape[1]) )
  return dataFrame

# This function will filter only some papers of the dataset
# Those that have a conection in the citations graph
def filter_paper_id( publications ):
  print( "Shape before "+str(publications.shape) )
  # Get files in the test data file
  good_papers = pd.read_csv( path_dataset+'/Paper_graph.csv' )
  good_papers.rename(columns = {'paper_id':'DOI'}, inplace = True)
  publications = pd.merge(publications,good_papers, how = 'inner', on=['DOI'])
  print( "Shape after "+str(publications.shape) )
  return publications

# This function will add the test data to the data (without ids of the authors)
def add_extra_data( publications ):
  print( "Shape before "+str(publications.shape) )
  # Get files in the test data file
  path = 'drive/MyDrive/Universidad/Tesis_sistema_de_recomendacion/Conjuntos_de_prueba'
  new_publications = pd.read_csv(path+'/test_set.csv')
  publications = publications.append(new_publications)
  publications.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
  publications = publications.drop_duplicates()
  print( "Shape after "+str(publications.shape) )
  return publications

# Make a join with the table of creation date of the papers
def add_creation_date( publications ):
  print( "Shape before "+str(publications.shape) )
  # Get file of creation dates
  dirname_target = 'drive/MyDrive/Universidad/Tesis/DATASETS'
  dates = pd.read_csv(dirname_target+'/Creation.csv')
  dates.rename(columns = {'paper_id':'DOI'}, inplace = True)
  # Make a join
  papers_with_dates = pd.merge(publications,dates, how = 'left', on=['DOI'])
  # Replace nulls
  papers_with_dates['created'] = papers_with_dates['created'].fillna('2000-01-01T00:00:00Z')
  print( "Shape after "+str(papers_with_dates.shape) )
  return papers_with_dates

# ============================ #
#   Make the relational model  #
# ============================ #

path_to_save_model = path_dataset
author_name = '/old_Author'
publication_name = '/old_Publication'
paper_name = '/old_Paper'
# structure to be used in the files of the relational model
paper_attributes = ['DOI', 'Title', 'Abstract', 'Author Keywords', 'Category']
paper_attributes_rename = ['paper_id','title','abstract','keywords','category']

def make_table_Author( df ):  
  # Iterate over the dataframe
  ans = set()
  for ids in df[ 'Author(s) ID' ]:
    idList = ids.split(';')
    for id in idList:
      if len(id)>0:        
        ans.add(str(id))
  
  # Convert the set to dataframe
  newDf = pd.DataFrame(list(ans))
  newDf.columns = ['author_id']
  # Delete nulls
  newDf = newDf.drop_duplicates()
  newDf = newDf.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)  
  newDf = newDf.drop_duplicates(subset=['author_id'])
  # Save the dataset
  newDf.to_csv(path_to_save_model+author_name+'.csv', index=False)

def make_table_Paper( df ):
  newDf = df[ paper_attributes ]  
  newDf.columns = paper_attributes_rename
  # Delete nulls
  newDf = newDf.drop_duplicates()
  newDf = newDf.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)  
  newDf = newDf.drop_duplicates(subset=['paper_id'])
  # Save the dataset
  newDf.to_csv(path_to_save_model+paper_name+'.csv', index=False)
  print( "Papers = "+str(newDf.shape) )

def make_table_publication( df ):
  # Empty dataframe
  newDf = pd.DataFrame({'author_id':[],'DOI':[]})

  # Iterate over the dataframe  
  publications = []
  for index, row in df.iterrows():
    ids = row['Author(s) ID']
    eid = row['DOI']
    idList = ids.split(';')
    for id in idList:
      if len(id)>0:
        publications.append( { 'author_id':id, 'paper_id':eid } )
    
  newDf = pd.DataFrame( publications )
  # Delete nulls
  newDf = newDf.drop_duplicates()
  newDf = newDf.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)  
  # Save the dataframe
  newDf.to_csv(path_to_save_model+publication_name+'.csv', index=False)
  print( str(newDf.shape[0]) + " " + str(newDf.shape[1]) )

def make_tables( df ):
  """
  This function will create the following relational model:

  Author( author_id )
  Publication( author_id, paper_id )
  Paper( paper_id, title, abstract, keywords, category )
  """

  # 532397 authors  
  make_table_Author(df)
  # papers 22867
  make_table_Paper(df)
  # publications 426470
  make_table_publication(df)
