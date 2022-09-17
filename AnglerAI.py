import numpy as np
import torch
import pyautogui

class AnglerAI(torch.nn.Module):
    def __init__(self):
        super().__init__()


        l1 = torch.nn.Linear(100, 20)
        r1 = torch.nn.ReLU()
        l2 = torch.nn.Linear(20, 5)
        r2 = torch.nn.ReLU()
        l3 = torch.nn.Linear(5, 1)
        sm = torch.nn.Softmax()

        self.seq = torch.nn.Sequential(
            l1,
            r1,
            l2,
            r2,
            l3,
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.sm(self.seq(x))


    def train_model(self, epochs=5, learning_rate=1e-3):
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
        train_loader = None

        n_total_steps = len(train_loader)

        for t in range(epochs):
            print(f"Epoch {t + 1}\n----------------------------------")
            train_loss = 0
            correct = 0
            for i, (X, y) in enumerate(train_loader):
                pred = self(X)
                train_loss = loss_fn(pred, y)

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

                if (i + 1) % 100 == 0:
                    print(f"\tEpoch {t + 1}, step {i + 1} / {n_total_steps}, loss = {train_loss.item():.4f}")


class AnglerDataset(torch.utils.data.Dataset):
    def __init__(self):
        xy = np.loadtxt("resources/fishing_data.csv", delimiter=",", dtype=np.float32, skiprows=0)
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, 0])

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.shape[0]
