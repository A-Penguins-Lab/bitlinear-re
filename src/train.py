import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

## Local imports
from models import SimpleModel, SimpleLinearModel

def load_dataloader():

    # Simple 1D input → 1D output
    X = torch.linspace(-5, 5, steps=1000).unsqueeze(1)  # Shape: (100, 1)
    y = 2 * X + 3 + torch.randn_like(X) * 0.2  # Add a bit of noise

    # Wrap into dataset
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

    return dataloader, X, y

def train(n_epochs: int):
    dataloader, X, y = load_dataloader()
    losses = []
    
    model = SimpleModel(X.shape[1], 1, 2)
    optim = torch.optim.Adam(model.parameters())
    loss_fn = nn.HuberLoss()
    
    for e in range(n_epochs):
        
        model.train()
        epoch_loss = 0.0

        for idx, (input, target) in enumerate(dataloader):
            
            out = model(input)
            loss = loss_fn(out, target)

            optim.zero_grad()
            loss.backward()
            optim.step()

            epoch_loss += loss.item()

        if e % 10 == 0:
            print(f'Loss at {e}', epoch_loss)

        losses.append(epoch_loss)


def eval(model):
    pass

if __name__ == "__main__":
    train(100)