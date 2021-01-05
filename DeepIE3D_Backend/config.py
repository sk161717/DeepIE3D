import math
#[n_iter:4, epsilon1, num_clusters200, k:0.5, std_num5000]..0.023223409352157293
#cluster graph parameter
model='Chair'
#edge_threshold=4
epsilon=1
num_clusters=200
n_iter=5
center=[0,0,0]
load_start=1
load_end=5
std_num=5000
k=0.5
candidate=13
isTuning=True
NG_border=0.75

def f(x):
    return 1/x

def selected(similarity):
    return 3*math.pow(similarity,5)

def unselected(similarity):
    return -3*math.pow(similarity,5)