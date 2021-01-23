import io,sys
from torch import Tensor
from collections import deque
import numpy as np
import torch
import binvox_rw
import networkx as nx
import math



def convert_array_to_graph(coords_arr,cube_len):
    
    coords_group = [[[None for k in range(cube_len)] for j in range(cube_len)] for i in range(cube_len)]#そのvoxelの所属するグループ
    vertice_number=[[[-1 for k in range(cube_len)] for j in range(cube_len)] for i in range(cube_len)]
    group_counts=[]
    graphs=[]
    group_id = 0
    for i in range(cube_len):
        for j in range(cube_len):
            for k in range(cube_len):
                if coords_group[i][j][k] == None and coords_arr[i][j][k]==True:
                    count,g=DFS(coords_arr, coords_group,vertice_number, group_id, i, j, k,cube_len)
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
    np_voxels=np.zeros((cube_len,cube_len,cube_len))
    for i in range(cube_len):
        for j in range(cube_len):
            for k in range(cube_len):
                if coords_group[i][j][k] ==max_index :
                    np_voxels[i][j][k]=1
    return max_graph,np_voxels

def return_label(i,j,k,cube_len,coords_arr=None,radius=None,search_area=None,index=None):
    is_distance_based=True
    label=0
    if is_distance_based:
        epsilon=1
        distance=np.linalg.norm(np.array([i-0,j,k-0])) #origin(0,32,0)
        scale=0.5/epsilon
        label=round(distance*scale)/scale
        return label
    else:    
        for a in range(i-search_area,i+search_area+1):
            for b in range(j-search_area,j+search_area+1):
                for c in range(k - search_area, k + search_area+1):
                    is_valid = 0 <= a < cube_len and 0 <= b < cube_len and 0 <= c < cube_len and coords_arr[a][b][c] == True
                    in_circle = np.linalg.norm(np.array([a-i,b-j,c-k]))<=radius
                    is_not_base= (a==i and b==j and c==k) ==False
                    if is_valid and in_circle and is_not_base:
                        label+=math.pow(2,index[a-i+1][b-j+1][c-k+1])
    return label

                        
def DFS(coords_arr, coords_group,vertice_number, group_id,i,j,k,cube_len):
    graph=nx.Graph()
    search_area=1
    is_circle=True
    radius=math.sqrt(2)
    #index=[[[None,0,None],[1,2,3],[None,4,None]],[[5,6,7],[8,None,9],[10,11,12]],[[None,13,None],[14,15,16],[None,17,None]]]
    #index=[[[None,6,None],[6,12,6],[None,6,None]],[[3,0,3],[0,None,0],[3,0,3]],[[None,9,None],[9,15,9],[None,9,None]]]
    index=[[[None,0,None],[0,0,0],[None,0,None]],[[0,0,0],[0,None,0],[0,0,0]],[[None,0,None],[0,0,0],[None,0,None]]]
    waiting=deque()
    vertice_to_x=[i]
    vertice_to_y=[j]
    vertice_to_z=[k]
    next_vertice_index=1
    coords_group[i][j][k]=group_id
    graph.add_node(0,label=0)
    graph.nodes[0]['label']=return_label(i,j,k,cube_len,coords_arr,radius,search_area,index)
    for a in range(i-search_area,i+search_area+1):
        for b in range(j-search_area,j+search_area+1):
            for c in range(k - search_area, k + search_area+1):
                is_valid = 0 <= a < cube_len and 0 <= b < cube_len and 0 <= c < cube_len and coords_arr[a][b][c] == True
                in_circle = (is_circle==False) or np.linalg.norm(np.array([a-i,b-j,c-k]))<=radius
                if is_valid and in_circle and coords_group[a][b][c]==None:
                    waiting.append(next_vertice_index)
                    coords_group[a][b][c]=-1*next_vertice_index
                    vertice_to_x.append(a)
                    vertice_to_y.append(b)
                    vertice_to_z.append(c)
                    graph.add_node(next_vertice_index,label=0)
                    graph.add_edge(0,next_vertice_index)
                    graph.nodes[next_vertice_index]['label']=return_label(a,b,c,cube_len,coords_arr,radius,search_area,index)
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
                    is_valid = 0 <= a < cube_len and 0 <= b < cube_len and 0 <= c < cube_len and coords_arr[a][b][c] == True
                    in_circle = (is_circle==False) or np.linalg.norm(np.array([a-i,b-j,c-k]))<=radius
                    if is_valid  and in_circle and (coords_group[a][b][c]==None or coords_group[a][b][c]<0):
                        if coords_group[a][b][c]==None:
                            waiting.append(next_vertice_index)
                            coords_group[a][b][c]=-next_vertice_index
                            vertice_to_x.append(a)
                            vertice_to_y.append(b)
                            vertice_to_z.append(c)   
                            graph.add_node(next_vertice_index,label=next_vertice_index)
                            graph.add_edge(cur_node,next_vertice_index)
                            graph.nodes[next_vertice_index]['label']=return_label(a,b,c,cube_len,coords_arr,radius,search_area,index)
                            next_vertice_index+=1
                        else:
                            waiting.append(coords_group[a][b][c]*-1)
                            graph.add_edge(cur_node,coords_group[a][b][c]*-1)
    return next_vertice_index,graph


    