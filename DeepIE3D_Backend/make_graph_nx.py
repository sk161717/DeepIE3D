import io,sys
from torch import Tensor
from collections import deque
import numpy as np
import torch
import binvox_rw
import networkx as nx



def convert_array_to_graph(coords_arr):
    
    coords_group = [[[None for k in range(64)] for j in range(64)] for i in range(64)]#そのvoxelの所属するグループ
    vertice_number=[[[-1 for k in range(64)] for j in range(64)] for i in range(64)]
    group_counts=[]
    graphs=[]
    group_id = 0
    for i in range(64):
        for j in range(64):
            for k in range(64):
                if coords_group[i][j][k] == None and coords_arr[i][j][k]==True:
                    count,g=DFS(coords_arr, coords_group,vertice_number, group_id, i, j, k)
                    group_counts.append(count)
                    graphs.append(g)
                    group_id += 1
    max_count=0
    max_graph=graphs[0]
    max_index=0
    for i in range(len(graphs)):
        if group_counts[i]>max_count:
            max_count=group_counts[i]
            max_graph=graphs[i]
            max_index=i
    np_voxels=np.zeros((64,64,64))
    for i in range(64):
        for j in range(64):
            for k in range(64):
                if coords_group[i][j][k] ==max_index :
                    np_voxels[i][j][k]=1
    return max_graph,np_voxels
                    
    
                        
def DFS(coords_arr, coords_group,vertice_number, group_id,i,j,k):
    graph=nx.Graph()
    search_area=1
    is_circle=True
    waiting=deque()
    vertice_to_x=[i]
    vertice_to_y=[j]
    vertice_to_z=[k]
    next_vertice_index=1
    coords_group[i][j][k]=group_id
    graph.add_node(0,label=0)
    for a in range(i-search_area,i+search_area+1):
        for b in range(j-search_area,j+search_area+1):
            for c in range(k - search_area, k + search_area+1):
                is_valid = 0 <= a < 64 and 0 <= b < 64 and 0 <= c < 64 and coords_arr[a][b][c] == True
                in_circle = (is_circle==False) or np.linalg.norm(np.array([a-i,b-j,c-k]))<=search_area
                if is_valid and in_circle and coords_group[a][b][c]==None:
                    waiting.append(next_vertice_index)
                    coords_group[a][b][c]=-1*next_vertice_index
                    vertice_to_x.append(a)
                    vertice_to_y.append(b)
                    vertice_to_z.append(c)
                    graph.add_node(next_vertice_index,label=0)
                    graph.add_edge(0,next_vertice_index)
                    next_vertice_index+=1             
    while len(waiting):
        cur_node=waiting.pop()
        if coords_group[vertice_to_x[cur_node]][vertice_to_y[cur_node]][vertice_to_z[cur_node]]!=None and coords_group[vertice_to_x[cur_node]][vertice_to_y[cur_node]][vertice_to_z[cur_node]]>=0:
            continue
        coords_group[vertice_to_x[cur_node]][vertice_to_y[cur_node]][vertice_to_z[cur_node]]=group_id
        i=vertice_to_x[cur_node]
        j=vertice_to_y[cur_node]
        k=vertice_to_z[cur_node]
        for a in range(i-search_area,i+search_area+1):
            for b in range(j-search_area,j+search_area+1):
                for c in range(k - search_area, k + search_area+1):
                    is_valid = 0 <= a < 64 and 0 <= b < 64 and 0 <= c < 64 and coords_arr[a][b][c] == True
                    in_circle = (is_circle==False) or np.linalg.norm(np.array([a-i,b-j,c-k]))<=search_area
                    if is_valid  and in_circle and (coords_group[a][b][c]==None or coords_group[a][b][c]<0):
                        if coords_group[a][b][c]==None:
                            waiting.append(next_vertice_index)
                            coords_group[a][b][c]=-next_vertice_index
                            vertice_to_x.append(a)
                            vertice_to_y.append(b)
                            vertice_to_z.append(c)   
                            graph.add_node(next_vertice_index,label=0)
                            graph.add_edge(cur_node,next_vertice_index)
                            next_vertice_index+=1
                        else:
                            waiting.append(coords_group[a][b][c]*-1)
                            graph.add_edge(cur_node,coords_group[a][b][c]*-1)
    return next_vertice_index,graph


    