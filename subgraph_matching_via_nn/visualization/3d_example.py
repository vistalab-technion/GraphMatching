# https://github.com/marcellodebernardi/loss-landscapes
import math

import numpy as np
import torch.nn
from loss_landscapes import random_plane
from loss_landscapes.metrics import Loss
from matplotlib import pyplot as plt
from torch import nn

steps = 80
dim = 500

# G, G_sub -> masked_G, G_sub -> masked_G_emb, G_sub_emb
# the localization module applies w on G to get masked_G. so the localization module is our model (w's mapped space is the parameters space)
# Given the two embeddings, we have a loss function (L2 loss for now, right?)
# What is the label? The GT label we want the model to predict is 0
  # x = (G, G_sub)
  # model (x):= (localization_func(w, x[0]), x[1])
  # loss:= return L2_loss(emb(x[0]), emb(x[1]))
  # Y = 0

x1 = torch.rand(dim)
x2 = torch.sin(2 * math.pi * x1) + 0.1 * torch.randn(dim)
X = torch.stack((x1, x2)).t()
y = torch.rand(dim).reshape(-1, 1)

loss_function = torch.nn.MSELoss()

class MyLinear(nn.Module):
  def __init__(self, in_features, out_features):
    super().__init__()
    self.weight = nn.Parameter(torch.randn(in_features, out_features))
    self.bias = nn.Parameter(torch.randn(out_features))

  def forward(self, input):
    return (input @ self.weight) + self.bias

model = MyLinear(X.shape[1], 1)

metric = Loss(loss_function, X, y)
landscape = random_plane(model, metric, normalization='filter', steps=steps)


# x = np.arange(-5,5,0.1)
# y = np.arange(-5,5,0.1)
# X,Y = np.meshgrid(x,y)
# Z = X*np.exp(-X - Y)

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')
# Plot a 3D surface
ax.plot_surface([i for i in range(steps)], [i for i in range(steps)], landscape)
plt.show()