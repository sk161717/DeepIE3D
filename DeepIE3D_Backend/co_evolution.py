import random
import numpy as np
import config as cfg
import math
import torch
import copy
from utils import generate_z
#配列で実装

class CoEvolution():
    def __init__(self):     
        self.pm=0.001
        self.N=200
        self.M=5
        self.part_num=int(self.N/self.M)
        self.pop_size=30
        self.select=9
        self.part_chrome_size=self.pop_size*self.N
                                    #[[instanceID,ID ..],[],..]
        self.history=8

        self.initialize()
    
    def initialize(self):
        self.population=[]
        self.past_population=[i for i in range(self.select)]
        self.past_selected=[]
        self.part_chrome=[PartChrome(self.M) for _ in range(self.part_chrome_size)]
        for i in range(self.pop_size):
            self.population.append([])
            for j in range(self.part_num):
                self.population[i].append(self.part_chrome[i*self.part_num+j])

    def return_initial(self,index):
        evolved_zs=[]
        for j in range(self.part_num):
                evolved_zs.extend(copy.deepcopy(self.population[index][j].chromosome))
        return evolved_zs

    
    def evolution(self,selected_canvases, mutation_rate=1.0):
        if len(selected_canvases)==0:
            self.past_population=[random.randrange(0,self.pop_size) for _ in range(self.pop_size)]
            return self.concat_population()
        self.refresh(selected_canvases)
        self.delete_history()
        self.child_reproduct(selected_canvases,mutation_rate)
        self.parent_reproduct(selected_canvases)
        self.select_next()
        return self.concat_population()

    def delete_history(self):
        while len(self.past_selected)>self.history:
            self.past_selected.pop(-1)
    
    def select_next(self):
        next_pop_index=[]
        while len(next_pop_index) < self.select:
            rand=random.randrange(0,self.pop_size)
            if rand not in next_pop_index:
                next_pop_index.append(rand)
        self.past_population=next_pop_index
        print('next index:',next_pop_index)

    def child_reproduct(self,selected_canvases,mutation_rate):
        for i in range(len(selected_canvases)):
            for j in range(self.part_num):
                parent1=self.population[selected_canvases[i]][j]
                parent2=self.population[selected_canvases[random.randrange(0,len(selected_canvases))]][random.randrange(0,self.part_num)]
                child1,child2=self.child_crossover(parent1,parent2)
                child1.mutate(mutation_rate)
                child2.mutate(mutation_rate)
                self.child_replace(child1)
                self.child_replace(child2)
                self.configure_relationship(parent1,parent2,child1,child2)


    def parent_reproduct(self,selected_canvases):
        if len(self.past_selected)<=1:
            return
        for i in range(len(selected_canvases)):
            parent1=self.population[self.past_selected[exclude_rand(0,len(self.past_selected),i)]]
            parent2=self.population[self.past_selected[i]]
            child1,child2=self.parent_crossover(parent1,parent2)
            self.parent_mutate(child1)
            self.parent_mutate(child2)
            self.parent_replace(child1)
            self.parent_replace(child2)


    def refresh(self,selected_canvases):
        for i in range(self.part_chrome_size):
            self.part_chrome[i].fitness=0
            self.part_chrome[i].child1=None
            self.part_chrome[i].child2=None
        for i in range(len(selected_canvases)):
            if self.past_population[selected_canvases[i]] not in self.past_selected:
                self.past_selected.insert(0,self.past_population[selected_canvases[i]])
            for j in range(self.part_num):
                self.population[selected_canvases[i]][j].fitness+=1
        

    def concat_population(self):
        evolved_zs=[]
        for i in range(self.select):
            evolved_zs.append([])
            index=self.past_population[i]
            for j in range(self.part_num):
                evolved_zs[i].extend(copy.deepcopy(self.population[index][j].chromosome))
        return evolved_zs


    def child_crossover(self,parent1,parent2):       #一点交叉
        child1=PartChrome(self.M)
        child2=PartChrome(self.M)
        parentcopy1=PartChrome(self.M)
        parentcopy2=PartChrome(self.M)
        cut=random.randrange(0,self.M+1)
        for i in range(self.M):
            if i<cut:
                child1.chromosome[i]=parent1.chromosome[i]
                child2.chromosome[i]=parent2.chromosome[i]
            else:
                child1.chromosome[i]=parent2.chromosome[i]
                child2.chromosome[i]=parent1.chromosome[i]
        parentcopy1.chromosome=copy.deepcopy(parent1.chromosome)
        parentcopy2.chromosome=copy.deepcopy(parent2.chromosome)
        rand=random.random()
        if rand<0.25:
            return parentcopy1,child1
        elif rand<0.5:
            return parentcopy1,child2
        elif rand<0.75:
            return parentcopy2,child1
        else:
            return parentcopy2,child2
        

    def child_replace(self,child):
        while(True):
            index=random.randrange(0,self.part_chrome_size)
            candidate=self.part_chrome[index]
            if candidate.fitness==0:
                candidate.chromosome=copy.deepcopy(child.chromosome)
                candidate.fitness=-1
                child=candidate
                #self.part_chrome[index]=child
                #child.fitness=-1
                break
    
    def parent_replace(self,child):
        while True:
            rand=random.randrange(0,self.pop_size)
            print('past select:',self.past_selected)
            if rand not in self.past_selected:
                print('replaced:',rand)
                self.population[rand]=child
                break

    def parent_crossover(self,parent1,parent2):         #一点交叉
        cut=random.randrange(0,self.part_num+1)
        child1=[]
        child2=[]
        for i in range(self.part_num):
            if i<cut:
                child1.append(parent1[i])
                child2.append(parent2[i])
            else:
                child1.append(parent2[i])
                child2.append(parent1[i])
        parentcopy1=copy.copy(parent1)
        parentcopy2=copy.copy(parent2)
        rand=random.random()
        if rand<0.25:
            return parentcopy1,child1
        elif rand<0.5:
            return parentcopy1,child2
        elif rand<0.75:
            return parentcopy2,child1
        else:
            return parentcopy2,child2
    
    def parent_mutate(self,child):
        for i in range(self.part_num):
            if random.random()<self.pm:
                child[i]=self.part_chrome[random.randrange(0,self.part_chrome_size)]  #mutation1
            if child[i].child1!=None and random.random()<0.5:
                if random.random()<0.5:
                    child[i]=child[i].child1                                        #mutation2
                else:
                    child[i]=child[i].child2


    def configure_relationship(self,parent1,parent2,child1,child2):
        parent1.child1=child1
        parent1.child2=child2
        parent2.child1=child1
        parent2.child2=child2



class PartChrome():
    def __init__(self,M):
        self.M=M
        self.chromosome=self.normal()
        self.fitness=0     #1..selected,0..no_selected,-1..child
        self.child1=None
        self.child2=None
       

    def normal(self,mutation_rate=1.0):
        return np.random.normal(0.0,mutation_rate,self.M).tolist()
    
    def mutate(self,mutation_rate):
        '''
        Mutate an individual
        '''
        
        mutation = self.benoulli().tolist()
        noise = self.normal(mutation_rate)#mutation_rate..分散。デフォルト->1.0は標準正規分布
        mutations = []
        for i in range(self.M):
            value = self.chromosome[i] + mutation[i] * noise[i]  #value=N(0,1)+(0,1)*N(0,1)
            if value > 5.0:
                value = 4.99
            if value < -5.0:
                value = -4.99
            mutations.append(value)
        self.chromosome=mutations

    def benoulli(self):
        '''
        The Benoulli distribution
        '''
        return torch.Tensor(self.M).bernoulli_(0.5)  #tensor([1., 1., 1., 0., 0., 1., 1., 0., 1.,..... 0., 0., 0., 1., 1., 1., 0., 1., 0.,])

def exclude_rand(start,end,exclude):
    while True:
        rand=random.randrange(start,end)
        if rand!=exclude:
            return rand


if __name__ == '__main__':
    EVO=CoEvolution()
    for i in  range(5):
        selected_canvases=[i for i in range(random.randrange(1,5))]
        evolved_zs=EVO.evolution(selected_canvases)
    print("OK")

