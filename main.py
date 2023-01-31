#!/usr/bin/env python
import random
from node import Node
from dataImport import *
# Main class
class NN:
    def __init__(self):
        # generate links with random weigths
        self.links=[
            self.generateLinks(784,16),
            self.generateLinks(16,16),
            self.generateLinks(16,10)]
        # generate nodes
        self.layers=[]
        self.layers=[
            self.generateNodes(784,0),
            self.generateNodes(16,1),
            self.generateNodes(16,2),
            self.generateNodes(10,3)
        ]
        self.data = openData()
        self.test(next(self.data))

    def generateNodes(self,count,index):
        # generate a list of nodes of length <count> with index <index>
        nodes = []
        for a in range(count):
            nodes.append(Node(bias=random.randint(1,10),index=(index,a),links=self.links,layers=self.layers))
        return nodes
    
    def generateLinks(self,inN,outN):
        # generates a matrix of random links
        col = []
        for a in range(inN):
            row = []
            for b in range(outN):   
                row.append(random.random())
            col.append(row)
        return col

    def test(self,dataset):
        label, image = dataset
        print(label)
        i=0
        for row in image:
            print(row)
            for data in row:
                self.layers[0][i].test(data)
                i+=1

    def learn(self):
        pass





if __name__ == "__main__":
    net = NN()
