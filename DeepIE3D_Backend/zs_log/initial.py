import csv
class Initial():
    def __init__(self):
        with open('initial_zs.csv','r') as f:
            reader=csv.reader(f)
            self.zs=[list(map(float,row)) for row in reader]
    def get_zs(self,index):
        return self.zs[index]
