import io,sys
from torch import Tensor
from collections import deque
import numpy as np
import torch
import binvox_rw
import igraph as ig


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
                    g=ig.Graph()
                    group_counts.append(DFS(coords_arr, coords_group,vertice_number, group_id, i, j, k,g))
                    graphs.append(g)
                    group_id += 1
    max_count=0
    max_graph=graphs[0]
    for i in range(len(graphs)):
        if group_counts[i]>max_count:
            max_count=group_counts[i]
            max_graph=graphs[i]
    return max_graph
                    
    
                        
def DFS(coords_arr, coords_group,vertice_number, group_id,i,j,k,graph):
    print('DFS start')
    search_area=2
    waiting=deque()
    vertice_to_x=[i]
    vertice_to_y=[j]
    vertice_to_z=[k]
    next_vertice_index=1
    coords_group[i][j][k]=group_id
    graph.add_vertices(1)
    for a in range(i-search_area,i+search_area+1):
        for b in range(j-search_area,j+search_area+1):
            for c in range(k - search_area, k + search_area+1):
                if 0 <= a < 64 and 0 <= b < 64 and 0 <= c < 64 and coords_arr[a][b][c] == True and coords_group[a][b][c]==None:
                    waiting.append(next_vertice_index)
                    coords_group[a][b][c]=-1*next_vertice_index
                    vertice_to_x.append(a)
                    vertice_to_y.append(b)
                    vertice_to_z.append(c)
                    graph.add_vertices(1)
                    graph.add_edges([(0,next_vertice_index)])
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
                    if 0 <= a < 64 and 0 <= b < 64 and 0 <= c < 64 and coords_arr[a][b][c] == True and (coords_group[a][b][c]==None or coords_group[a][b][c]<0):
                        if coords_group[a][b][c]==None:
                            waiting.append(next_vertice_index)
                            coords_group[a][b][c]=-next_vertice_index
                            vertice_to_x.append(a)
                            vertice_to_y.append(b)
                            vertice_to_z.append(c)   
                            graph.add_vertices(1)
                            graph.add_edges([(cur_node,next_vertice_index)])
                            next_vertice_index+=1
                        else:
                            waiting.append(coords_group[a][b][c]*-1)
                            graph.add_edges([(cur_node,coords_group[a][b][c]*-1)])
                            
    return next_vertice_index

'''
    print('OK')
    
    coords_group[i][j][k] = group_id
    vertice_number[i][j][k]=group_count
    if len(graph.vs)<=group_count:
        graph.add_vertices(1)
    group_count += 1
    group_count_const=group_count
    print(i,j,k)
    for a in range(i-search_area,i+search_area+1):
        for b in range(j-search_area,j+search_area+1):
            for c in range(k - search_area, k + search_area+1):   
                if 0 <= a < 64 and 0 <= b < 64 and 0 <= c < 64 and coords_arr[a][b][c] == True and coords_group[a][b][c] == -1:
                    graph.add_edges([(group_count_const-1,vertice_number[a][b][c])])
    for a in range(i-search_area,i+search_area+1):
        for b in range(j-search_area,j+search_area+1):
            for c in range(k - search_area, k + search_area+1):     
                if 0 <= a < 64 and 0 <= b < 64 and 0 <= c < 64 and coords_arr[a][b][c] == True and coords_group[a][b][c] == group_id:
                    if (group_count_const-1)!=vertice_number[a][b][c]:
                        print(a,b,c,group_count_const,i)
                        graph.add_edges([(group_count_const-1,vertice_number[a][b][c])])
                elif 0 <= a < 64 and 0 <= b < 64 and 0 <= c < 64 and coords_arr[a][b][c] == True and coords_group[a][b][c] == -1:
                    group_count=DFS(coords_arr, coords_group,vertice_number, group_id, a, b, c,group_count,graph)
'''
    