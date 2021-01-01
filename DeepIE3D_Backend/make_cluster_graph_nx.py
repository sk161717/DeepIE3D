import io,sys
from torch import Tensor
from collections import deque
import numpy as np
import torch
import binvox_rw
import networkx as nx
import math
import config as cfg
from operator import itemgetter

def nodes_connected(u, v, G):
    return u in G.neighbors(v)

def distance_matrix(coords,n):
    tmp_index=np.arange(n)
    xx, yy = np.meshgrid(tmp_index, tmp_index)
    distances=np.linalg.norm(coords[xx]-coords[yy], axis=2)
    return distances

def return_label(coord,center):
    epsilon=cfg.epsilon
    distance=np.linalg.norm(coord-center) #origin(0,32,0)
    scale=0.5/epsilon
    label=round(distance*scale)/scale
    return label

def convert_sparse_to_graph(coords,center):   #coords..[float,float,float]*N,input numpy
    n=len(coords)
    distance=distance_matrix(coords,n)
    graph=nx.Graph()
    for i in range(n):
        graph.add_node(i,label=return_label(coords[i],center))
    for i in range(n):
        tuple_list=list(enumerate(distance[i]))
        sort_tuple=sorted(tuple_list,key=itemgetter(1))
        D=sort_tuple[1][1]
        for j in range(1,n):
            if sort_tuple[j][1]<(1+cfg.k*cfg.f(j))*D:
                if nodes_connected(i,sort_tuple[j][0],graph)==False:
                    graph.add_edge(i,sort_tuple[j][0])
            else:
                break
    return graph

    
