import time
import torch

def train(model, train_loader, optimizer, criterion, epoch, device):
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
    print('Epoch: {} | Loss: {:.3f} | Train time (min): {}'.format(
        epoch, train_loss, (time_end - time_start) / 60))