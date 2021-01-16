import random
import torch
import numpy as np
import config as cfg
import math
import grakel
import binvox_rw
from utils import generate_z
from torch import Tensor
from make_cluster_graph_nx import convert_sparse_to_graph
from operator import itemgetter
from sklearn.cluster import KMeans

# Constants
FOREIGN = 0
ELITE=3
PROPOSE=9
PAST=1
Z_SIZE = 200

class Evolution():
    def __init__(self):
        self.past_graphs=[]
        self.NG_graphs=[]


    def WL_evolution(self,selected_canvases,NG_canvases, zs,G,D, novelty=False, behavioral=False, mutation_rate=1.0):
        if len(selected_canvases)==0 and len(NG_canvases)==0:
            return simple_evolution(selected_canvases, zs,G)
        elif len(selected_canvases)==0:
            candidate_zs = []
            candiate_voxels,NG_voxels=[],[]
            for i in range(PROPOSE):
                if i in NG_canvases:
                    NG_voxels.append(G.generate(torch.Tensor(zs[i]),cfg.model))
            self.generate_candidate(selected_canvases,zs,None,candidate_zs,candiate_voxels,mutation_rate,G)
            candidate_graphs=cluster_graph(candiate_voxels)   #軽量化
            NG_graphs=cluster_graph(NG_voxels) 

            self.NG_graphs+=NG_graphs
            graphs=candidate_graphs+self.NG_graphs

            G_grakel=grakel.graph_from_networkx(graphs, node_labels_tag='label')
            gk=grakel.WeisfeilerLehman(n_iter=cfg.n_iter,normalize=True)
            K,_l=gk.fit_transform(G_grakel)

            ordered_index=judgeNG(K)
            evolved_zs=[]
            for i in range(PROPOSE):
                evolved_zs.append(candidate_zs[ordered_index[i]])
            return evolved_zs
        else:
            selected_zs,candidate_zs = [], []
            selected_voxels,candiate_voxels=[],[]
            print('selectedCanvas:',selected_canvases)

            for i in range(PROPOSE):
                if i in selected_canvases:
                    selected_zs.append(zs[i])
                    selected_voxels.append(G.generate(torch.Tensor(zs[i]), cfg.model))
            
            self.generate_candidate(selected_canvases,zs,selected_zs,candidate_zs,candiate_voxels,mutation_rate,G)
        
            selected_graphs=cluster_graph(selected_voxels)
            candidate_graphs=cluster_graph(candiate_voxels)
                        
            graphs=candidate_graphs+selected_graphs   #next gen,past gen, before past generation,NG

            G_grakel=grakel.graph_from_networkx(graphs, node_labels_tag='label')
            gk=grakel.WeisfeilerLehman(n_iter=cfg.n_iter,normalize=True)
            K,_=gk.fit_transform(G_grakel)
            print(K)
            evolved_zs=[]
            ordered_index=self.predict_fitness(K,len(selected_zs),mutation_rate)

            evolved_zs.append(candidate_zs[0])
            for i in range(PROPOSE-PAST):
                evolved_zs.append(candidate_zs[ordered_index[i]])
            
            return evolved_zs

    def predict_fitness(self,K,selected_length,mutation_rate=1.0):    
        if selected_length>=2:
            if mutation_rate>=0.5:
                mutate_good_index=select_good(K,PAST,PAST+cfg.mutate_gen_2expl,PAST+cfg.candidate_2expl,cfg.mutate_select_2expl)
                cross_good_index=select_good(K,PAST+cfg.mutate_gen_2expl,PAST+cfg.mutate_gen_2expl+cfg.cross_gen_2expl,PAST+cfg.candidate_2expl,cfg.cross_select_2expl)
                randcross_good_index=select_good(K,PAST+cfg.mutate_gen_2expl+cfg.cross_gen_2expl,PAST+cfg.candidate_2expl,PAST+cfg.candidate_2expl,cfg.randcross_select_2expl)
                cross_good_index.extend(randcross_good_index)
                mutate_good_index.extend(cross_good_index)
                print("2 expl result:",mutate_good_index)
                return mutate_good_index
            else:
                mutate_good_index=select_good(K,PAST,PAST+cfg.mutate_gen_2conv,PAST+cfg.candidate_2conv,cfg.mutate_select_2conv)
                cross_good_index=select_good(K,PAST+cfg.mutate_gen_2conv,PAST+cfg.mutate_gen_2conv+cfg.cross_gen_2conv,PAST+cfg.candidate_2conv,cfg.cross_select_2conv)
                randcross_good_index=select_good(K,PAST+cfg.mutate_gen_2conv+cfg.cross_gen_2conv,PAST+cfg.candidate_2conv,PAST+cfg.candidate_2conv,cfg.randcross_select_2conv)
                cross_good_index.extend(randcross_good_index)
                mutate_good_index.extend(cross_good_index)
                print("2 conv result:",mutate_good_index)
                return mutate_good_index
    
        elif selected_length==1:
            if mutation_rate>=0.5:
                mutate_good_index=select_good(K,PAST, PAST+cfg.mutate_gen_1expl,PAST+cfg.candidate_1expl,cfg.mutate_select_1expl)
                randcross_good_index=select_good(K,PAST+cfg.mutate_gen_1expl,PAST+cfg.candidate_1expl,PAST+cfg.candidate_1expl,cfg.randcross_select_1expl)
                mutate_good_index.extend(randcross_good_index)
                print("mutate_good_index {} randcross_good_index {}".format(mutate_good_index,randcross_good_index))
                print("1 expl result:",mutate_good_index)
                return mutate_good_index
            else:
                mutate_good_index=select_good(K,PAST, PAST+cfg.mutate_gen_1conv,PAST+cfg.candidate_1conv,cfg.mutate_select_1conv)
                randcross_good_index=select_good(K,PAST+cfg.mutate_gen_1conv,PAST+cfg.candidate_1conv,PAST+cfg.candidate_1conv,cfg.randcross_select_1conv)
                mutate_good_index.extend(randcross_good_index)
                print("1 conv result:",mutate_good_index)
                return mutate_good_index

            
    
    def generate_candidate(self,selected_canvases,zs,selected_zs,candidate_zs,candidate_voxels,mutation_rate,G):
        if len(selected_canvases)>=2 :
            candidate_zs.append(zs[selected_canvases[0]])
            if mutation_rate >= 0.5:
                assert cfg.mutate_select_2expl+cfg.cross_select_2expl+cfg.randcross_select_2expl==PROPOSE-PAST,"invalid parameter 2 explore"
                candidate_zs.extend([random_mutate(selected_zs, mutation_rate) for _ in range(cfg.mutate_gen_2expl)])
                candidate_zs.extend([simple_crossover(selected_zs) for _ in range(cfg.cross_gen_2expl)])
                candidate_zs.extend([random_crossover_with_random(selected_zs) for _ in range(cfg.randcross_gen_2expl)])
                candidate_voxels.extend([G.generate(torch.Tensor(candidate_zs[i]),cfg.model) for i in range(PAST+cfg.candidate_2expl)])
            else:
                assert cfg.mutate_select_2conv+cfg.cross_select_2conv+cfg.randcross_select_2conv==PROPOSE-PAST,"invalid parameter 2 converge"
                candidate_zs.extend([random_mutate(selected_zs, mutation_rate) for _ in range(cfg.mutate_gen_2conv)])
                candidate_zs.extend([simple_crossover(selected_zs) for _ in range(cfg.cross_gen_2conv)])
                candidate_zs.extend([random_crossover_with_random(selected_zs) for _ in range(cfg.randcross_gen_2conv)])
                candidate_voxels.extend([G.generate(torch.Tensor(candidate_zs[i]),cfg.model) for i in range(PAST+cfg.candidate_2conv)])
        elif len(selected_canvases)==1:
            candidate_zs.append(zs[selected_canvases[0]])
            if mutation_rate >=0.5:
                assert cfg.mutate_select_1expl+cfg.randcross_select_1expl==PROPOSE-PAST,"invalid parameter 1 explore"
                candidate_zs.extend([mutate(selected_zs[0], mutation_rate) for _ in range(cfg.mutate_gen_1expl)])
                candidate_zs.extend([crossover_with_random(selected_zs[0]) for _ in range(cfg.randcross_gen_1expl)])
                candidate_voxels.extend([G.generate(torch.Tensor(candidate_zs[i]),cfg.model) for i in range(PAST+cfg.candidate_1expl)])
            else:
                assert cfg.mutate_select_1conv+cfg.randcross_select_1conv==PROPOSE-PAST,"invalid parameter 1 converge"
                candidate_zs.extend([mutate(selected_zs[0], mutation_rate) for _ in range(cfg.mutate_gen_1conv)])
                candidate_zs.extend([crossover_with_random(selected_zs[0]) for _ in range(cfg.randcross_gen_1conv)])
                candidate_voxels.extend([G.generate(torch.Tensor(candidate_zs[i]),cfg.model) for i in range(PAST+cfg.candidate_1conv)])
        else:
            candidate_zs.extend([normal().tolist() for _ in range(cfg.candidate)])
        
        return

def select_good(K,candidate_start,candidate_end,all_candidate,selected_num):
    print("candidate_start {},candidate_end {},all_candidate {},selected_num {}".format(candidate_start,candidate_end,all_candidate,selected_num))
    fitnesses=[] #fitnesses=[amax(K[0][selected]),amax(K[1][selected]),..]
    for i in range(candidate_start, candidate_end):
        similarity_array=K[i][all_candidate:]  
        n=len(similarity_array)
        fitnesses.append(np.amax(similarity_array))
        print('n:',n,' sim_array:',similarity_array)
    tuple_list=list(enumerate(fitnesses))
    sort_tuple=sorted(tuple_list,key=itemgetter(1),reverse=True)
    print('sorted:',sort_tuple)
    ordered_index=[sort_tuple[i][0] for i in range(selected_num)]
    ordered_index=np.array(ordered_index)+candidate_start
    print("ordered index:",ordered_index)
    return ordered_index.tolist()



def judgeNG(K):
    ordered_index=[]
    for i in range(cfg.candidate):
        similarity_array=K[i][cfg.candidate:]
        n=len(similarity_array)
        if np.amax(similarity_array)<cfg.NG_border:
            ordered_index.append(i)
        print('n:',n,' sim_array:',similarity_array)
    random.shuffle(ordered_index)
    print("ordered:",ordered_index)
    return ordered_index[:PROPOSE]


def cluster_graph(voxels,std_num=None,num_clusters=None,k=None,epsilon=None):
    if std_num==None:
        std_num=cfg.std_num
    if num_clusters==None:
        num_clusters=cfg.num_clusters
    graphs=[]
    for i in range(len(voxels)):
        nonzero=torch.nonzero(voxels[i]>=0.5)
        count=torch.count_nonzero(nonzero)
        index=nonzero.to('cpu').detach().numpy()
        num_clusters_=int(num_clusters*math.sqrt(count/std_num))
        print("num_clusters:",num_clusters_)
        km = KMeans(n_clusters=num_clusters_,
            init='random',
            n_init=2,
            max_iter=10,
            tol=1e-04,
            random_state=0
            )
        km.fit_predict(index)
        cluster_centers=km.cluster_centers_
        graph=convert_sparse_to_graph(cluster_centers,np.array(cfg.center),k,epsilon)
        graphs.append(graph)
    return graphs
        
def make_binvox(data,filename):
    nonzero=torch.nonzero(data>=0.5)
    dense_data=binvox_rw.sparse_to_dense(nonzero.to('cpu').detach().numpy().T,[64,64,64])
    voxels=binvox_rw.Voxels(dense_data,[64,64,64], [0,0,0], 1, 'xyz')
    voxels.write('../DeepIE3D_Backend/made/'+filename+cfg.model+'.binvox')


def simple_evolution(selected_canvases, zs, G, novelty=False, behavioral=False, mutation_rate=1.0):
    '''
    1:1 implementation of DeepIE if not [novelty], else DeepIE with added novelty search
    '''
    selected_zs, evolved_zs = [], []

    for i in selected_canvases:
        selected_zs.append(zs[i])

    delta = 9 - len(selected_zs)
    x = max(0, delta - FOREIGN)
    if selected_zs:
        evolved_zs.extend([mutate(simple_crossover(selected_zs), mutation_rate)
                        for _ in range(x)])
    else:
        if novelty:
            return behavioral_novelty_search([], G) if behavioral else novelty_search([])
        else:
            return [normal().tolist() for _ in range(9)]
    print("1len(zs):" , len(evolved_zs))
    x = min(FOREIGN, delta)
    evolved_zs.extend([mutate(selected_zs[i], mutation_rate)
                    for i in range(len(selected_zs))])
    print("2len(zs):",len(evolved_zs))
    if novelty:
        evolved_zs = behavioral_novelty_search(
            [], G) if behavioral else novelty_search(evolved_zs)
    else:
        evolved_zs.extend([normal().tolist() for _ in range(x)])
    print("3len(zs):", len(evolved_zs))
    #e_zs_np = np.array(evolved_zs)
    #zero_np = np.zeros((55, 200))
    #e_pooled_np = np.concatenate([e_zs_np, zero_np,e_zs_np,zero_np])
    #print("size:", e_pooled_np.shape)
    '''fake = G.g_chair(Tensor(e_zs_np))
    print("fake size:",fake.size())
    print(D.discriminate(fake))'''
    return evolved_zs



def simple_crossover(population):
    '''
    Crossover used in DeepIE - picks randomly two from the population
    '''
    a = population[random.randint(0, len(population)-1)]
    b = population[random.randint(0, len(population)-1)]
    return crossover(a, b)

def mirror_simple_crossover(population):
    a = population[random.randint(0, len(population)-1)]
    b = population[random.randint(0, len(population)-1)]
    return mirror_crossover(a,b)

def random_mutate(population,mutation_rate=1.0):
    rand=random.randint(0, len(population)-1)
    return mutate(population[rand],mutation_rate)

def random_crossover_with_random(population):
    a = population[random.randint(0, len(population)-1)]
    return crossover_with_random(a)


def crossover(a, b):         #[-2.420189619064331, 1.28584885597229,....]
    '''
    Crossover two genes
    '''
    mask = benoulli().tolist()
    return [mask[i] * a[i] + (1 - mask[i]) * b[i] for i in range(Z_SIZE)]

def mirror_crossover(a,b):
    mask=benoulli().tolist()
    return [mask[i] * a[i] + (1 - mask[i]) * b[i] for i in range(Z_SIZE)],[mask[i] * b[i] + (1 - mask[i]) * a[i] for i in range(Z_SIZE)]

def crossover_with_random(a):
    random_z=generate_z().tolist()
    mask=benoulli().tolist()
    return [mask[i] * a[i] + (1 - mask[i]) * random_z[i] for i in range(Z_SIZE)]

def mirror_crossover_with_random(a):
    random_z=generate_z().tolist()
    mask=benoulli().tolist()
    return [mask[i] * a[i] + (1 - mask[i]) * random_z[i] for i in range(Z_SIZE)],[mask[i] * random_z[i] + (1 - mask[i]) * a[i] for i in range(Z_SIZE)]

def benoulli():
    '''
    The Benoulli distribution
    '''
    return torch.Tensor(Z_SIZE).bernoulli_(0.5)  #tensor([1., 1., 1., 0., 0., 1., 1., 0., 1.,..... 0., 0., 0., 1., 1., 1., 0., 1., 0.,])


def normal(mutation_rate=1.0):
    '''
    The normal distribution
    '''
    return torch.Tensor(Z_SIZE).normal_(0.0, mutation_rate)


def mutate(individual, mutation_rate=1.0):
    '''
    Mutate an individual
    '''
    mutation = benoulli().tolist()
    noise = normal(mutation_rate).tolist()#mutation_rate..分散。デフォルト->1.0は標準正規分布
    mutations = []
    for i in range(Z_SIZE):
        value = individual[i] + mutation[i] * noise[i]  #value=N(0,1)+(0,1)*N(0,1)
        if value > 5.0:
            value = 4.99
        if value < -5.0:
            value = -4.99
        mutations.append(value)

    return mutations

def mutate_mask(individual,mask, mutation_rate=1.0):
    '''
    Mutate an individual
    '''

    noise = normal(mutation_rate).tolist()#mutation_rate..分散。デフォルト->1.0は標準正規分布
    mutations = []
    for i in range(Z_SIZE):
        value = individual[i] + mask[i] * noise[i]  #value=N(0,1)+(0,1)*N(0,1)
        if value > 5.0:
            value = 4.99
        if value < -5.0:
            value = -4.99
        mutations.append(value)

    return mutations


def novel_number(numbers):
    '''
    Given a list of numbers, find the number furthest apart from all other numbers
    '''
    numbers.append(4.896745231)
    numbers.append(-4.987654321)
    numbers.sort()
    distance = -1.0
    number = 0.5
    for i in range(1, len(numbers)):
        temp_distance = abs(numbers[i] - numbers[i-1])
        if temp_distance > distance:
            distance = temp_distance
            number = numbers[i-1]
    return number + distance/2.0


def create_novel(zs):
    '''
    Create a single novel Z vector based on previous Z vectors
    '''
    novel = []
    for i in range(Z_SIZE):
        numbers = []
        for z in zs:
            numbers.append(z[i])
        novel.append(novel_number(numbers))
    return novel


def novelty_search(zs):
    '''
    Perform novelty search on the Z vectors
    '''
    if not zs:
        zs.append(normal().tolist())
    novel_size = 9 - len(zs)
    for _ in range(novel_size):
        zs.append(create_novel(zs))

    return zs


def behavioral_novelty_search(zs, G, n=10):
    '''
    Perform behavioral novelty search - noveltry metric: voxel intersections
    '''
    models = []
    if not zs:
        zs.append(normal().tolist())
    for z in zs:
        models.append(torch.round(G.generate(torch.Tensor(z), 'Chair')))
    while len(zs) < 9:
        n_zs, n_models = generate_models(G, n)
        similarity = 100
        z_index = -1
        for index, model in enumerate(n_models):
            temp_similarity = get_sim(model, models)
            if temp_similarity < similarity:
                similarity = temp_similarity
                z_index = index
        zs.append(n_zs[z_index].tolist())
        models.append(torch.round(G.generate(n_zs[z_index], cfg.model)))
    return zs


def generate_models(G, n):
    '''
    Generates [n] z vectors and [n] 3D models
    '''
    zs = [normal() for _ in range(n)]
    models = [torch.round(G.generate(zs[i], cfg.model)) for i in range(n)]
    return zs, models


def get_sim(model, dataset):
    '''
    Returns similarity of a model compared to a dataset
    '''
    size = len(dataset)
    model = torch.nonzero(model)
    voxel_count = len(model)
    same_voxel_count = 0
    for voxels in dataset:
        for (x, y, z) in model:
            if voxels[x][y][z]:
                same_voxel_count += 1
    return same_voxel_count/size/voxel_count*100

def process_discriminate(candidate_zs,G,D):
    for i in range(len(candidate_zs)):
        fake = G.g_chair(Tensor(candidate_zs[i]))
        #concatenated=torch.cat((fake,fake),1).to('cpu')
        #print("discriminator:",D.discriminate(concatenated))

def process_base_kernel(base_kernel):
    for i in range(len(base_kernel)):
        print('shape',base_kernel[i].X.shape)
        print(np.dot(base_kernel[i].X,base_kernel[i].X.T))

def custom_evolve(selected_canvases, zs, G, novelty=False, behavioral=False, mutation_rate=1.0):
    selected_zs, evolved_zs = [], []

    for i in selected_canvases:
        selected_zs.append(zs[i])
    
    if len(selected_canvases)==0:
        return [normal().tolist() for _ in range(9)]
    elif len(selected_canvases)==1:
        evolved_zs.append(selected_zs[0])
        evolved_zs.extend([crossover_with_random(selected_zs[0]) for _ in range(6)])
        evolved_zs.extend([x for x in mirror_crossover_with_random(selected_zs[0])])
    elif len(selected_canvases)==2:
        evolved_zs.extend(selected_zs)
        evolved_zs.extend([x for x in mirror_crossover(selected_zs[0],selected_zs[1])])
        evolved_zs.extend([mutate(selected_zs[0], mutation_rate)
                    for i in range(6)])
    return evolved_zs


