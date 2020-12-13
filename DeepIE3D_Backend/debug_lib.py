import WL as wl
import igraph as ig
import make_graph
import binvox_rw
import numpy as np
import torch
import networkx as nx
import grakel
import matplotlib.pyplot as plt
import make_graph_nx
import math
import random
import csv
import torch.nn as nn
from IPython.display import SVG, display
from mpl_toolkits.mplot3d import axes3d, Axes3D
from generate import SuperGenerator
from evolution import mutate
from utils import generate_z,slide
from make_graph_nx import return_label

def WeisfeilerLehman(voxels,n,_n_iter):
    trimed_voxels=[]
    G=[]
    for i in range(n):
        Gi,trimed_voxeli=make_graph_nx.convert_array_to_graph(voxels[i])
        G.append(Gi)
        trimed_voxels.append(trimed_voxeli)
    G_grakel=grakel.graph_from_networkx(G, node_labels_tag='label')
    gk=grakel.WeisfeilerLehman(n_iter=_n_iter,normalize=True)
    K=gk.fit_transform(G_grakel)
    return K,trimed_voxels,G

def load_file_data(container,n):
    for i in range(9,n+9,1):
        with open ('../DeepIE3D_Training/data/chair64_512/chair'+str(i)+'.binvox','rb') as f:
            model=binvox_rw.read_as_3d_array(f)
            container.append(model.data)
    return container

voxels=[]
n=2
load_file_data(voxels,n)
#K,trimed_voxels,G=WeisfeilerLehman(voxels,n,3)
#print(K)
input_=torch.randn(1,1,64,64,64)
m=nn.MaxPool3d(2,stride=2)
output=m(input_)
print("OK")
