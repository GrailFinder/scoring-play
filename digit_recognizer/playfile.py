import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

train_df = pd.read_csv("train.csv")

print(train_df.info())
print(train_df.shape)

random_sel = np.random.randint(len(train_df), size=8)

grid = make_grid(torch.Tensor((
    train_df.iloc[random_sel, 1:].as_matrix()
    /255.).reshape((-1, 28, 28))).unsqueeze(1), nrow=8)

plt.imshow(grid.numpy().transpose((1,2,0)))
plt.show()