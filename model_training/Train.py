import time
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

def train(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    epoch: int,
    device: torch.device,
    save_every: int,
    idx_to_word: dict
):
  model.train()
  train_loss = 0

  time_start = time.time()
  for batch in train_loader:
    contexts, targets = batch
    contexts = [context.to(device) for context in contexts]
    targets = [target.to(device) for target in targets]
    # inputs, targets = inputs.to(device), targets.to(device)
    optimizer.zero_grad()

    for context, target in zip(contexts, targets):
      output = model(context)
      target = torch.tensor([target], dtype=torch.long, device=device)
      loss = criterion(output, target)
      train_loss += loss.item()

      loss.backward()
    

    optimizer.step()


  time_end = time.time()
  print('Epoch: {} | Loss: {:.3f} | Train time (min): {:.4f}'.format(
      epoch, train_loss, (time_end - time_start) / 60.0))

  # Saving the model
  if epoch % save_every == 0:
    torch.save(model.state_dict(), 'model.pt')
    model.save_embeddings('embeddings.txt', idx_to_word)


def train_multi_gpu(gpu_id: int,
                    model: torch.nn.Module,
                    train_loader: torch.utils.data.DataLoader,
                    optimizer: torch.optim.Optimizer,
                    criterion: torch.nn.Module,
                    save_every: int,
                    epoch: int,
                    idx_to_word: dict
                    ):
  # Setting up the DistributedDataParallel model
  model = model.to(gpu_id)
  model = DDP(model, device_ids=[gpu_id])
  train_loader.set_epoch(epoch)

  model.train()
  train_loss = 0

  time_start = time.time()
  for batch in train_loader:
    contexts, targets = batch
    contexts = [context.to(gpu_id) for context in contexts]
    targets = [target.to(gpu_id) for target in targets]
    # inputs, targets = inputs.to(device), targets.to(device)
    optimizer.zero_grad()

    for context, target in zip(contexts, targets):
      output = model(context)
      target = torch.tensor([target], dtype=torch.long, device=gpu_id)
      loss = criterion(output, target)
      train_loss += loss.item()

      loss.backward()
    

    optimizer.step()


  time_end = time.time()
  print('[GPU {}] -> Epoch: {} | Loss: {:.3f} | Train time (min): {:.4f}'.format(
      gpu_id, epoch, train_loss, (time_end - time_start) / 60.0))
  
  # Saving the model
  if epoch % save_every == 0 and gpu_id == 0:
    torch.save(model.module.state_dict(), 'model.pt')
    model.module.save_embeddings('embeddings.txt', idx_to_word)