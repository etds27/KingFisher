import os

import numpy as np
import torch
import pyautogui


class AnglerAI(torch.nn.Module):
    def __init__(self):
        super().__init__()

        l1 = torch.nn.Linear(300, 20)
        r1 = torch.nn.ReLU()
        l2 = torch.nn.Linear(20, 5)
        r2 = torch.nn.ReLU()
        l3 = torch.nn.Linear(5, 1)

        self.seq = torch.nn.Sequential(
            l1,
            r1,
            l2,
            r2,
            l3,
        )

    def forward(self, x):
        # print(x.shape)
        # x = self.flatten(x)
        return self.seq(x).type(torch.FloatTensor)

    def train_model(self, epochs=5, learning_rate=1e-3):
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
        train_dataset = AnglerDataset()
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=64,
                                                   shuffle=True)

        n_total_steps = len(train_loader)

        for t in range(epochs):
            print(f"Epoch {t + 1}\n----------------------------------")
            train_loss = 0
            for i, (X, y) in enumerate(train_loader):
                # print(X.shape)
                pred = self(X)
                y = y.unsqueeze(1)
                optimizer.zero_grad()
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                if (i + 1) % 41 == 0:
                    print(f"\tEpoch {t + 1}, step {i + 1} / {n_total_steps}, loss = {train_loss:.4f}")


    def catch_fish(self):
        pass


    def click(self, press_time):
        pyautogui.mouseDown()
        pyautogui.sleep(press_time)
        pyautogui.mouseUp()

class AnglerDataset(torch.utils.data.Dataset):
    """
    Data set will be a collection of the following parameters repeated for the last x intervals

    Will need to play with intervals
    6 data points per time snapshot
    Need to test different input sizes
    looking for previous 5 seconds

    interval:
        0.01s = 500 * 5 = 2500
        0.025 = 250 * 5 = 1250
        0.05s = 100 * 5 = 500
        0.1s = 50 * 5 = 250

    params
        fish_location
        bar_upper_pct
        bar_lower_pct
        progress
        total click time between interval
    """
    def __init__(self):
        self.x = np.array([])
        self.y = np.array([])
        total_xy = None

        for i, f in enumerate(os.listdir("training_data_25/prepared")):
            #print(f)
            if not i:
                total_xy = np.loadtxt(os.path.join("training_data_25/prepared", f), delimiter=",", dtype=np.float32, skiprows=0)
            else:
                xy = np.loadtxt(os.path.join("training_data_25/prepared", f), delimiter=",", dtype=np.float32, skiprows=0)
                # print(xy.shape)
                total_xy = np.concatenate((total_xy, xy))


        #xy = np.loadtxt("resources/fishing_data.csv", delimiter=",", dtype=np.float32, skiprows=0)
        self.x = torch.from_numpy(total_xy[:, 1:])
        self.y = torch.from_numpy(total_xy[:, 0]).type(torch.FloatTensor)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.shape[0]


if __name__ == "__main__":
    # dataset = AnglerDataset()

    ai = AnglerAI()
    ai.train_model(100)
