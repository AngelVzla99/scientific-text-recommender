# paths 
main_path = 'drive/MyDrive/Universidad/Tesis_sistema_de_recomendacion'
path_to_embeddings = main_path+'/Embeddings'
path_to_embeddings_graph = main_path+'/Embeddings_graph'
path_to_dataset = main_path+'/Dataset'
path_to_test_set = main_path+'/Conjuntos_de_prueba'
path_to_results = main_path+'/Resultados'
path_to_MLP = main_path+'/MLP_models'

from sklearn.metrics import ndcg_score
import sys
import torch
from torch.utils.data import Dataset
from torch import nn
sys.path.append( '../Data_manipulation' )
import datasets_api 

class CustomImageDataset(Dataset):
  def __init__(self, api, dataset_name, transform=None, target_transform=None):        
    self.x_dataset, self.y_dataset = api.get_dataset(dataset_name)
    self.transform = transform
    self.target_transform = target_transform

  def __len__(self):
    return len(self.y_dataset)

  def __getitem__(self, idx):
    embedding = self.x_dataset[idx]
    labels = self.y_dataset[idx]
    if self.transform:
        image = self.transform(image)
    if self.target_transform:
        label = self.target_transform(label)
    return torch.from_numpy(embedding), torch.from_numpy(labels)


torch.manual_seed(7)
torch.set_default_tensor_type(torch.DoubleTensor)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self, l1 = 16):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(1168, 25802)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

def train(dataloader, model, loss_fn, optimizer):
  size = len(dataloader.dataset)
  model.train()
  for batch, (X, y) in enumerate(dataloader):
    X, y = X.to(device), y.to(device)

    # Compute prediction error
    pred = model(X)
    loss = loss_fn(pred, y)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # if batch % 100 == 0:
    #     loss, current = loss.item(), (batch + 1) * len(X)
    #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
  size = len(dataloader.dataset)
  num_batches = len(dataloader)
  model.eval()
  test_loss, correct = 0, 0
  with torch.no_grad():
      for X, y in dataloader:
          X, y = X.to(device), y.to(device)
          pred = model(X)
          test_loss += loss_fn(pred, y).item()
          correct += (pred.argmax(1) == y).type(torch.float).sum().item()
  test_loss /= num_batches
  correct /= size
  # print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
  return test_loss

def test_nDCG(dataloader, model, loss_fn):
  # Eval model
  size = len(dataloader.dataset)
  num_batches = len(dataloader)
  model.eval()
  test_loss, correct = 0, 0
  with torch.no_grad():
    for X, y in dataloader:
      X, y = X.to(device), y.to(device)
      pred = model(X)

      # Convert to numpy
      pred = pred.cpu().numpy()
      y = y.cpu().numpy()

      test_loss += ndcg_score(y, pred)

  test_loss /= num_batches
  correct /= size
  # print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
  return test_loss

class gatem:
  model_name = None
  model = None
  api = None
  device = None 
  
  def __init__(self, model_name, text_embedder, graph_embedder):
    # api to get data
    self.api = datasets_api( text_embedder, graph_embedder )
    self.model_name = 'test'
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # DELETE THIS, IT'S FOR TESTING PURPOSE
    self.model = torch.load(path_to_MLP+'/'+model_name+'.pth', map_location=torch.device(self.device) )
    self.model.eval()
  
  def sort_and_split(self,results):  
    # retult = [ [similarity,doi] ]
    # Sort results    
    results.sort(); results.reverse()
    # Answers 
    ans_dois = [ x[1] for x in results ]
    ans_similarity = [ x[0] for x in results ]
    return ans_dois, ans_similarity

  def get_recommendation( self, doi ):
    # Get prediction
    doi_representation = self.api.get_representation( doi )
    doi_representation = torch.from_numpy(doi_representation).to(self.device)#.to(torch.float32)
    with torch.no_grad():
      model_prediction = self.model( doi_representation )
      # sort recommendations
      n = model_prediction.shape[0]
      recommendation = []
      for i in range(n):
        if self.api.id_to_doi(i) != doi:
          recommendation.append([ model_prediction[i].item(), self.api.id_to_doi(i) ])
      return self.sort_and_split( recommendation ) 

  def get_recommendation_V2( self, doi ):
    # Get prediction
    doi_representation = self.api.get_representation( doi )
    doi_representation = torch.from_numpy(doi_representation).to(self.device)#.to(torch.float32)
    with torch.no_grad():
      model_prediction = self.model( doi_representation )
      # sort recommendations
      n = model_prediction.shape[0]
      recommendation = []
      ans_dois = [None]*n
      ans_similarity = [None]*n
      for i in range(n):
        ans_dois[i] = self.api.id_to_doi(i) 
        ans_similarity[i] = model_prediction[i].item() 
      return ans_dois, ans_similarity