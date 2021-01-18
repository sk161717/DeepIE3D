import io,sys
from torch import Tensor
import numpy as np
import torch
import binvox_rw


def generate_z(z_size=200, dis='norm1',seed=None):
    """
    Generates a latent vector with a given distribution.
    Can be seeeded for test
    """
    if not seed is None:
        torch.manual_seed(seed)
    if dis == 'norm1':
        z=Tensor(z_size).normal_(0.0, 1.0)
        '''
        with open("z_counter.txt",mode='r+') as f:
            index=int(f.read())  
            z[0]=(index%9)*0.2-1
            print(index)
            f.write(str(index+1))
            '''
        return z
        
    elif dis == 'norm033':
        return Tensor(z_size).normal_(0.0, 0.33)
    elif dis == 'uni':
        return torch.rand(z_size)


def create_coords_from_voxels(voxels):
    """
    Creates a nonzero lsit of coords from voxels
    """
    nonzero = torch.nonzero(voxels >= 0.5)   #index list
    return nonzero.tolist()

def create_coords_from_voxels_generate(voxels):
    sys.setrecursionlimit(100000)
    nonzero = torch.nonzero(voxels >= 0.5)
    return eliminate_isolated(nonzero).tolist()

def eliminate_isolated(coords):
    coords_arr = [[[0 for k in range(64)] for j in range(64)] for i in range(64)]
    coords_group = [[[-1 for k in range(64)] for j in range(64)] for i in range(64)]
    group_counts=[]
    group_id = 0
    for i in range(coords.shape[0]):
        coords_arr[coords[i][0]][coords[i][1]][coords[i][2]] = 1
    for i in range(64):
        for j in range(64):
            for k in range(64):
                if coords_group[i][j][k] == -1 and coords_arr[i][j][k]==1:
                    group_counts.append(DFS(coords_arr, coords_group, group_id, i, j, k, 0))
                    group_id += 1
    coords_np=np.empty([0,3])
    for i in range(64):
        for j in range(64):
            for k in range(64):
                if group_counts[coords_group[i][j][k]] < 30 and coords_arr[i][j][k] == 1:
                    coords_arr[i][j][k] = 0
                if coords_arr[i][j][k] == 1: 
                    coords_np=np.append(coords_np,[[i, j, k]],axis=0)
    coords_tensor=torch.tensor(coords_np)
    return coords_tensor
                    
    
                        
def DFS(coords_arr, coords_group, group_id, i, j, k,group_count):
    search_area=1
    coords_group[i][j][k] = group_id
    group_count += 1
    for a in range(i-search_area,i+search_area+1):
        for b in range(j-search_area,j+search_area+1):
            for c in range(k - search_area, k + search_area):     
                if 0 <= a < 64 and 0 <= b < 64 and 0 <= c < 64 and coords_arr[a][b][c] == 1 and coords_group[a][b][c] == -1:
                    group_count=DFS(coords_arr, coords_group, group_id, a, b, c,group_count)
    return group_count
                

def calculate_camera(voxelss):
    """
    Calculates min and max x,y,z based on several 3D models
    """
    min_x, min_y, min_z, max_x, max_y, max_z = 1337, 1337, 1337, -1337, -1337, -1337
    for voxels in voxelss:
        coords = create_coords_from_voxels(voxels)
        x_min, y_min, z_min, x_max, y_max, z_max = find_extremas(coords)
        min_x = x_min if x_min < min_x else min_x
        min_y = y_min if y_min < min_y else min_y
        min_z = z_min if z_min < min_z else min_z
        max_x = x_max if x_max > max_x else max_x
        max_y = y_max if y_max > max_y else max_y
        max_z = z_max if z_max > max_z else max_z

    return min_x, min_y, min_z, max_x, max_y, max_z


def find_extremas(coords):
    """
    Find the extremas in a 3D array
    """
    min_x, min_y, min_z, max_x, max_y, max_z = 1337, 1337, 1337, -1337, -1337, -1337
    for coord in coords:
        min_x = coord[0] if coord[0] < min_x else min_x
        min_y = coord[1] if coord[1] < min_y else min_y
        min_z = coord[2] if coord[2] < min_z else min_z
        max_x = coord[0] if coord[0] > max_x else max_x
        max_y = coord[1] if coord[1] > max_y else max_y
        max_z = coord[2] if coord[2] > max_z else max_z

    return min_x, min_y, min_z, max_x, max_y, max_z


def generate_binvox_file(voxels):
    """
    Given  a 3D tensor, create a binvox file
    """
    data = voxels.numpy()
    size = len(data[0])
    for x in range(size):
        for y in range(size):
            for z in range(size):
                data[x][y][z] = 1.0 if data[x][y][z] > 0.5 else 0.0
    size = len(voxels[0])
    dims = [size, size, size]
    scale = 1.0
    translate = [0.0, 0.0, 0.0]
    output = io.BytesIO()
    model = binvox_rw.Voxels(data, dims, translate, scale, 'xyz')
    model.write(output)
    output.seek(0)
    return output

def slide(voxels,step):
    m=64
    voxels_slide= [[[False for k in range(m)] for j in range(m)] for i in range(m)]
    for i in range(0,m):
        for j in range(0,m):
            for k in range(0,m):
                if voxels[i][j][k]==True:
                    if 0<=i+step<m and 0<=j+step<m and 0<=k+step<m:
                        voxels_slide[i+step][j+step][k+step]=True
                    else:
                        print("slide over the range")
    return voxels_slide