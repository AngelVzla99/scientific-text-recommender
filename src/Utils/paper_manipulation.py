from sklearn.metrics.pairwise import cosine_similarity


def take_df_from_column( df_papers_from, column_name, columns_value ):
  '''
  df_papers_from is a dataframe with the collection of papers
  column_name is a string with the name of the category
  '''
  # making boolean series for a team name
  filter = df_papers_from[column_name]==columns_value
  df_ans = df_papers_from.where(filter)
  return df_ans.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)  
  
def print_paper_info(papers,pos,max_len=20):
  print(papers.iloc[pos]['title'])
  print(papers.iloc[pos]['category'])
  print(papers.iloc[pos]['keywords'])
  print("\n")

  number_of_words = 0
  for word in papers.iloc[pos]['abstract'].split(' '):
    print(word, end=" ")
    number_of_words += 1
    if number_of_words == max_len:
      print("")
      number_of_words = 0
  print("")

def print_top_n( N, papers, embeddingsA, embeddingsB, pos_paper, list_of_rows=[] ):
  # Similarity between articles
  M = cosine_similarity( embeddingsA, embeddingsB )
  base_category = papers.iloc[pos_paper]['category']
  # Top n articles
  n = len(M[0]) if type(M[0])==list else M.shape[1]
  ans = []
  for j in range(n):

    valid_category = False
    if len(list_of_rows)>0:
      for r in list_of_rows:
        if r == j:
          valid_category = True
    else: 
      valid_category = True


    if j!=pos_paper and valid_category:
      ans.append([ M[pos_paper][j], pos_paper, j ])
  # Sorting
  ans.sort(); ans.reverse()
  top_n = [ ans[i] for i in range(N) ]     
  # Answer about more similar articles
  for val in top_n:
    ans = val[0]; pos_i = val[1]; pos_j = val[2]
    print("\n==== Paper "+str(pos_j+1)+" ====")
    print( str(ans) + " " + str(pos_i) + " "+ str(pos_j) )
    print_paper_info(papers,pos_j)  
  