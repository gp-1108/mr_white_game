from SentencesDataset import SentencesDataset
from CBOW import CBOW
from Train import train
from torch.utils.data import DataLoader
import torch
import argparse

def collate_func(batch):
  contexts = []
  targets = []
  for i in range(0, len(batch)):
    contexts += batch[i][0]
    targets += batch[i][1]
  return contexts, targets


def main(
    dataset_path: str,
    context_size: int,
    embedding_dim: int,
    epochs: int,
    batch_size: int,
    save_every: int
):

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
      model,
      dataloader,
      optimizer,
      criterion,
      epoch,
      device,
      save_every,
      dataset.idx_to_word)
  
  # Saving the model
  print('Saving the model...')
  torch.save(model.state_dict(), 'model.pt')
  model.save_embeddings('embeddings.txt', dataset.idx_to_word)
  

if __name__ == '__main__':
  parser = argparse.ArgumentParser("Distributed Data Parallel Training")
  parser.add_argument('-d', '--dataset_path', type=str, default='dataset_manipulation/it_polished.txt')
  parser.add_argument('-c', '--context_size', type=int, default=2)
  parser.add_argument('-e', '--embedding_dim', type=int, default=300)
  parser.add_argument('-ep', '--epochs', type=int, default=10)
  parser.add_argument('-b', '--batch_size', type=int, default=32)
  parser.add_argument('-s', '--save_every', type=int, default=5)

  main_args = (
    parser.parse_args().dataset_path,
    parser.parse_args().context_size,
    parser.parse_args().embedding_dim,
    parser.parse_args().epochs,
    parser.parse_args().batch_size,
    parser.parse_args().save_every,
  )

  print(main_args)
  main(*main_args)