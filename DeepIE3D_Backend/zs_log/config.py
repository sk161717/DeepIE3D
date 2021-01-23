import math
#[n_iter:4, epsilon1, num_clusters200, k:0.5, std_num5000]..0.023223409352157293
#cluster graph parameter
model='Chair'
#edge_threshold=4
#epsilon=2
epsilon=1
#num_clusters=50
num_clusters=200
#n_iter=3
n_iter=5
center=[0,0,0]
load_start=1
load_end=5
std_num=5000
k=0.5
candidate=30
NG_border=0.63

explore_mutation_rate=1.0
converge_mutation_rate=0.4


#selected 1 and explore
randcross_gen_1expl=6
randcross_select_1expl=4
mutate_gen_1expl=7
mutate_select_1expl=4
candidate_1expl=randcross_gen_1expl+mutate_gen_1expl

#selected 1 and converge
randcross_gen_1conv=0
randcross_select_1conv=0
mutate_gen_1conv=11
mutate_select_1conv=8
candidate_1conv=randcross_gen_1conv+mutate_gen_1conv

#selected >=2 and explore
randcross_gen_2expl=10
randcross_select_2expl=3
mutate_gen_2expl=3
mutate_select_2expl=2
cross_gen_2expl=5
cross_select_2expl=3
candidate_2expl=randcross_gen_2expl+mutate_gen_2expl+cross_gen_2expl

#selected >=2 and converge
randcross_gen_2conv=0
randcross_select_2conv=0
mutate_gen_2conv=5
mutate_select_2conv=3
cross_gen_2conv=7
cross_select_2conv=5
candidate_2conv=randcross_gen_2conv+mutate_gen_2conv+cross_gen_2conv

def f(x):
    return 1/x

def selected(similarity):
    return 3*math.pow(similarity,5)

def unselected(similarity):
    return -3*math.pow(similarity,5)