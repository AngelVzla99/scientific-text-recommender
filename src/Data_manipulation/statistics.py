# !pip install pandasql
# !pip install SQLAlchemy==1.*

# Paths
main_path = 'drive/MyDrive/Universidad/Tesis_sistema_de_recomendacion'
path_test_set = main_path+'/Conjuntos_de_prueba'
path_dataset = main_path+'/Dataset'

import pandas as pd
from pandasql import sqldf
import matplotlib.pyplot as plt

class Statistics():
  def __init__(self, Author, Publication, Paper):
    self.Author = Author
    self.Publication = Publication
    self.Paper = Paper

  def print_publication_cnt(self):
    pubTable = self.Publication
    pubTable = sqldf("SELECT author_id, COUNT(*) as pub_count FROM pubTable GROUP BY author_id", env=None)

    #print("Global average of publications")
    #print( sqldf("SELECT AVG(pub_count) FROM pubTable") )

    df_total = pd.DataFrame()
    for val in range(2,30):
      df = sqldf("SELECT "+str(val)+" as cantidad,  COUNT(*) as cnt FROM pubTable WHERE pub_count="+str(val), env=None)
      df_total = pd.concat([df_total,df])
    
    ##This get the bar plot
    df_total.plot(x='cantidad',y='cnt',kind='bar')
    plt.show()
    
  def plot_publications_ge(self):
    pubTable = self.Publication
    pubTable = sqldf("SELECT author_id, COUNT(*) as pub_count FROM pubTable GROUP BY author_id")

    df_total = pd.DataFrame()
    for val in range(2,30):
      df = sqldf("SELECT "+str(val)+" as cantidad,  COUNT(*) as cnt FROM pubTable WHERE pub_count>="+str(val))      
      df_total = df_total.append(df)
    
    ##This get the bar plot
    df_total.plot(x='cantidad',y='cnt',kind='bar')
    plt.show()

  def histogram_categories(self):
    paperTable = self.Paper
    df = sqldf("SELECT category, COUNT(*) as cnt FROM paperTable GROUP BY category ORDER BY cnt ASC LIMIT 20")

    display(df)

    ##This get the bar plot
    df.plot(x='category',y='cnt',kind='bar')
    plt.show()

  def histogram_multidisciplinary(self):
    paperTable = self.Paper
    publicationTable = self.Publication
    df = sqldf("\
      SELECT numCat, COUNT(*) as cnt \
      FROM \
      (\
        SELECT author_id, COUNT(*) as numCat \
        FROM \
        ( \
          SELECT DISTINCT author_id, category \
          FROM paperTable \
          NATURAL JOIN publicationTable \
        ) \
        GROUP BY author_id\
      )\
      GROUP BY numCat\
      ")

    ##This get the bar plot
    print(df.head())
    df.plot(x='numCat',y='cnt',kind='bar')
    plt.show()

  def iterator_group_by(self):
    pubTable = self.Publication
    pubTable = sqldf("SELECT * FROM pubTable ORDER BY author_id")

    accum = []
    last = ""
    for i, row in pubTable.iterrows():
      id = row['author_id']
      eid = row['EID']
      if id==last:
        accum.append(eid)
      else:
        if last!="": yield (id,accum)
        last = id; accum = [eid]
    yield (id,accum)
    
  def publications_group_by(self):
    i = 0
    for (id,accum) in self.iterator_group_by():
      print( str(id) + " " + str(accum) )
      # Break after some iterations
      i+=1
      if i==20: break
