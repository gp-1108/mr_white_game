from SentencesDataset import SentencesDataset
from CBOW import CBOW
from Train import train
from torch.utils.data import DataLoader
import torch
import sys

def collate_func(batch):
  contexts = []
  targets = []
  for i in range(0, len(batch)):
    contexts += batch[i][0]
    targets += batch[i][1]
  return contexts, targets


def main(
    dataset_path='dataset_manipulation/it_polished.txt',
    context_size=2,
    embedding_dim=300,
    epochs=10,
    batch_size=32,
):
  # Parsing int arguments
  context_size = int(context_size)
  embedding_dim = int(embedding_dim)
  epochs = int(epochs)
  batch_size = int(batch_size)

  # Creating the dataset
  print('Creating the dataset...')
  dataset = SentencesDataset(dataset_path, context_size)
  dataset.print_info()

  # Creating the dataloader
  print('Creating the dataloader...')
  dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    collate_fn=collate_func
  )

  # Creating the model
  print('Creating the model...')
  model = CBOW(len(dataset.word_to_idx), embedding_dim, context_size)

  # Training the model
  print('Training the model...')
  optimizer = torch.optim.Adam(model.parameters())
  criterion = torch.nn.NLLLoss()
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model.to(device)
  for epoch in range(epochs):
    train(
      model, dataloader, optimizer, criterion, epoch, device)
  
  # Saving the model
  print('Saving the model...')
  torch.save(model.state_dict(), 'model.pt')
  model.save_embeddings('embeddings.txt', dataset.idx_to_word)
  

if __name__ == '__main__':
  args = sys.argv[1:]
  main(*args)