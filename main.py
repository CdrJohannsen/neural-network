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
        self.inLayer = self.generateNodes(784)
        self.hLayer = self.generateNodes(16)
        self.hLayer2 = self.generateNodes(16)
        self.oLayer = self.generateNodes(10)
        self.data = openData()

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

    def learn(self):
        pass





if __name__ == "__main__":
    net = NN()
    net.learn()
