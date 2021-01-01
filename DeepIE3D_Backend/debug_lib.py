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

def WeisfeilerLehman(voxels,n,_n_iter,cube_len):
    trimed_voxels=[]
    G=[]
    for i in range(n):
        Gi,trimed_voxeli=make_graph_nx.convert_array_to_graph(voxels[i],cube_len)
        G.append(Gi)
        trimed_voxels.append(trimed_voxeli)
    G_grakel=grakel.graph_from_networkx(G, node_labels_tag='label')
    gk=grakel.WeisfeilerLehman(n_iter=_n_iter,normalize=True)
    K,base_kernel=gk.fit_transform(G_grakel)
    print(base_kernel)
    return K,trimed_voxels,G

def load_file_data(container,n):
    for i in range(11-n,11,1):
        with open ('../DeepIE3D_Training/data/chair64_512/chair'+str(i)+'.binvox','rb') as f:
            model=binvox_rw.read_as_3d_array(f)
            container.append(model.data)
    return container

def make_pooled_binvox(data,filename,dim):
    voxels=binvox_rw.Voxels(data,[dim,dim,dim], [0,0,0], 1, 'xyz')
    voxels.write('../DeepIE3D_Training/data/ex/'+filename+'_'+str(dim)+'.binvox')

def maxpool(voxels,kernel_size):
    pooled_voxels=[]
    for i in range(len(voxels)):
        tensor=torch.Tensor(voxels[i]).unsqueeze_(0).unsqueeze_(0)
        m=nn.MaxPool3d(kernel_size,stride=kernel_size)
        out=m(tensor).squeeze_(0).squeeze_(0)
        make_pooled_binvox(out.to('cpu').detach().numpy(),'pooled'+str(i),64/kernel_size)
        pooled_voxels.append(out)
    return pooled_voxels

voxels=[]
n=2
cube_len=16
load_file_data(voxels,n)
#voxels32=maxpool(voxels,2)
voxels16=maxpool(voxels,int(64/cube_len))
#K32,trimed_voxels32,G32=WeisfeilerLehman(voxels32,n,3,32)
K16,trimed_voxels16,G16=WeisfeilerLehman(voxels16,n,3,cube_len)
print(K16)
