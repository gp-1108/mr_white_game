import time
import torch

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

