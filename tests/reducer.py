import path_setup
import torch
import torch.nn as nn
import torch.nn.functional as F
from VaeDimReducerJ.VaeDimReducerJ import VaeReducer
from VaeDimReducer.VaeDimReducer import VaeDimReducer
import numpy as np
from pathlib import Path

from sklearn.datasets import load_iris, load_breast_cancer, load_digits
import plotly.express as px
from torchvision import datasets
import torchvision.transforms as transforms


import pandas as pd
data_path = Path(path_setup.root / 'tests' / 'data')
# Path.mkdir(data_path, exist_ok=True)

transform = transforms.Compose([transforms.ToTensor(),])
# Load our FashionMNIST dataset
trainset = datasets.FashionMNIST(
    root=data_path,
    train=True, 
    download=True,
    transform=transform)

testset = datasets.FashionMNIST(
    root=data_path,
    train=False,
    download=True,
    transform=transform)


X = trainset.data.flatten(start_dim=1) / 255.0
X = X[::20]
# X = (X - X.mean()) / X.std()
y = trainset.targets
y = y[::20]
classes = trainset.classes
labels = [classes[yi] for yi in y]

# X, y = load_iris(return_X_y=True)
# X, y = load_breast_cancer(return_X_y=True)
# X, y = load_digits(return_X_y=True)
# labels = y


# reducer = VaeDimReducer(input_dim=X.shape[-1], latent_dim=2, encoder_layers=[40]*2, decoder_layers=[40]*2, minR2Score=0.9, device='cuda')
# reducer.fit(X)
# Z = reducer.transform(X)
# if not isinstance(Z, np.ndarray):
#     Z = Z.cpu().numpy()
# fig = px.scatter()
# fig.add_scatter(x=Z[:,0], y=Z[:,1], mode='markers', marker=dict(color=y, size=5, colorscale='Rainbow'), hovertext=labels,hoverinfo="text") #colorscale='rainbow'
# fig.update_layout(title="2D representation", plot_bgcolor="black", paper_bgcolor="black", font=dict(color="white"),
#                                                      legend=dict(bordercolor='white', borderwidth=1))
# fig.show()

reducer2 = VaeReducer(input_dim=X.shape[-1], latent_dim=2, encoder_layers=[80]*3, decoder_layers=[80]*3, target_r2=0.9, beta=0.0001, max_epochs=700, device='cuda', max_patiance=300, valid_size=0.15)
reducer2.fit(X)
Z = reducer2.transform(X)
if not isinstance(Z, np.ndarray):
    Z = Z.cpu().numpy()
fig = px.scatter()
fig.add_scatter(x=Z[:,0], y=Z[:,1], mode='markers', marker=dict(color=y, size=5, colorscale='Rainbow'), hovertext=labels,hoverinfo="text") #colorscale='rainbow'
fig.update_layout(title="2D representation", plot_bgcolor="black", paper_bgcolor="black", font=dict(color="white"),
                                                     legend=dict(bordercolor='white', borderwidth=1))
fig.show()

# from sklearn.manifold import Isomap

# oIsomap  = Isomap(n_neighbors=6, n_components=2, metric='l2').fit(X)
# mZ       = oIsomap.transform(X)
# fig = px.scatter()
# fig.add_scatter(x=mZ[:,0], y=mZ[:,1], mode='markers', marker=dict(color=y, size=5, )) #colorscale='rainbow'
# fig.update_layout(title="2D representation", plot_bgcolor="black", paper_bgcolor="black", font=dict(color="white"),
#                                                      legend=dict(bordercolor='white', borderwidth=1))
# fig.show()


print('done')