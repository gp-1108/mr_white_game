import os
import torch
from torch.utils.data import Dataset

class SentencesDataset(Dataset):
  def __init__(self, txt_path, context_size):
    print("Current working directory:", os.getcwd())
    self._load_data(txt_path)
    self.context_size = context_size

  def __len__(self):
    return len(self.sentences)

  def __getitem__(self, idx):
    sentence = self.sentences[idx]
    contexts = []
    targets = []
    for i in range(self.context_size, len(sentence) - self.context_size):
      context = (
        [sentence[i-j] for j in range(self.context_size, 0, -1)] +
        [sentence[i+j] for j in range(1, self.context_size+1, 1)]
      )
      target = sentence[i]
      context, target = self._to_torch_tensors(context, target)
      contexts.append(context)
      targets.append(target)
    return contexts, targets

  def _to_torch_tensors(self, context, target):
    context = torch.tensor([self.word_to_idx[word] for word in context])
    target = torch.tensor(self.word_to_idx[target])
    return context, target

  def _load_data(self, txt_path):
    with open(txt_path, 'r') as f:
      self.sentences = f.read().splitlines()

    self.sentences = [s.split() for s in self.sentences]

    self.word_to_idx = {}
    self.idx_to_word = {}

    idx = 0
    for sentence in self.sentences:
      for word in sentence:
        if word not in self.word_to_idx:
          self.word_to_idx[word] = idx
          self.idx_to_word[idx] = word
          idx += 1

  def from_idx_to_words(self, idxs: list):
    words = [self.idx_to_word[idx.item()] for idx in idxs]
    return words

  def print_info(self):
    print('Dataset info:')
    print('--Number of sentences: {}'.format(len(self.sentences)))
    print('--Vocabulary size: {}'.format(len(self.word_to_idx)))
