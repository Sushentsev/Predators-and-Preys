import torch
import torch.nn as nn
import argparse
import sys
sys.path.append("/home/denis/Study/HSE/Predators-and-Preys")
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from os import listdir
import numpy as np
from tqdm import tqdm


class ActorNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)



def get_data_part(path: str):
    file_names = listdir(path)
    ids = np.array(list(set([int(file_name.split(":")[0]) for file_name in file_names])))

    indices = np.random.permutation(np.arange(len(ids)))
    ids = ids[indices]

    for id in ids:
        action_file_name = str(id) + ":actions.npy"
        state_file_name = str(id) + ":states.npy"

        X = torch.FloatTensor(np.load(f"{path}/{state_file_name}"))
        y = torch.FloatTensor(np.load(f"{path}/{action_file_name}"))

        yield X, y


def train(model: ActorNetwork, num_epochs: int, data_path: str, save_path: str):
    optimizer = Adam(params=model.parameters(), weight_decay=1e-4)
    mse_loss = nn.MSELoss()

    for epoch in range(num_epochs):
        total_loss = 0.
        num_parts = 0

        for X, y in tqdm(get_data_part(data_path)):

            num_batches = 0
            batch_total_loss = 0.
            train_dataloader = DataLoader(TensorDataset(X, y), batch_size=512, shuffle=True)

            for X_batch, y_batch in train_dataloader:
                optimizer.zero_grad()

                logits = model(X_batch)
                loss = mse_loss(logits, y_batch)
                loss.backward()
                optimizer.step()

                num_batches += 1
                batch_total_loss += loss.item()

            total_loss += (batch_total_loss / num_batches)
            num_parts += 1

        print(f"Epoch: {epoch}")
        print(f"Average MSE loss: {round(total_loss / num_parts, 4)}")
        torch.save(model.state_dict(), save_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--state_dim", type=int)
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--num_epochs", default=50, type=int)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--save_path", type=str)
    params = parser.parse_args()

    model = ActorNetwork(state_dim=params.state_dim, hidden_dim=params.hidden_dim)
    train(model, num_epochs=params.num_epochs, data_path=params.data_path, save_path=params.save_path)
    print(f"Model successfully saved!")