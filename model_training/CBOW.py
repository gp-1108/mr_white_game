import torch.nn as nn
import torch.nn.functional as F
import os

class CBOW(nn.Module):
  def __init__(self, vocab_size, embedding_dim, context_size, gpu_id=0):
    super(CBOW, self).__init__()
    self.embeddings = nn.Embedding(vocab_size, embedding_dim)
    self.linear1 = nn.Linear(2 * context_size * embedding_dim, 128)
    self.linear2 = nn.Linear(128, vocab_size)
    self.gpu_id = gpu_id
  
  def forward(self, contexts):
    embeds = self.embeddings(contexts).view((1, -1))
    out = F.relu(self.linear1(embeds))
    out = self.linear2(out)
    log_probs = F.log_softmax(out, dim=1)
    return log_probs
  
  def save_embeddings(self, file_path, id2word):
    """
    Save the word embeddings to a file.

    Args:
    - file_path (str): The file path where the embeddings will be saved.
    - id2word (dict): A dictionary mapping word indices to their corresponding words.
    """
    # Remove old embeddings if already present
    if os.path.exists(file_path):
      os.remove(file_path)

    embeddings = self.embeddings.weight.data.cpu().numpy()
    with open(file_path, 'w', encoding='utf-8') as f:
      for idx, word in id2word.items():
        embedding = ' '.join(map(str, embeddings[idx]))
        f.write(f'{word} {embedding}\n')