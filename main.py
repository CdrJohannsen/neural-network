#!/usr/bin/env python
import random
from node import Node
from dataImport import *

class NN:
    def __init__(self):
        self.links=[
            self.generateLinks(784,16),
            self.generateLinks(16,16),
            self.generateLinks(16,10)]
        self.layers=[
            self.generateNodes(784),
            self.generateNodes(16),
            self.generateNodes(16),
            self.generateNodes(10)
        ]
        self.data = openData()
        self.test(next(self.data))

    def generateNodes(self,count):
        nodes = []
        for a in range(count):
            nodes.append(Node(bias=random.randint(1,10),index=a,links=self.links))
        return nodes
    
    def generateLinks(self,inN,outN):
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
