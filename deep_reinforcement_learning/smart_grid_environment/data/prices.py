import torch
import pandas as pd
import numpy as np

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class EnergyPrices():
    def __init__(self):
        self.prices_frame = pd.read_csv('smart_grid_environment/data/prices.csv', decimal=',')

    def __len__(self):
        return len(self.prices_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # if isinstance(idx, slice):
        #     print("Getting prices from", self.prices_frame["date"].values[idx.start], "to", self.prices_frame["date"].values[idx.stop])

        price = np.array(self.prices_frame["price"].values[idx], dtype=float)

        return price

    def max_price(self):
        return self.prices_frame["price"].max()
