#!/usr/bin/env python
import random
from node import Node
from dataImport import *

class NN:
    def __init__(self):
        self.inLayer = self.generateNodes(784)
        self.hLayer = self.generateNodes(16)
        self.hLayer2 = self.generateNodes(16)
        self.data = openData()

    def generateNodes(self,count):
        nodes = []
        for a in range(count):
            nodes.append(Node(bias=random.randint(1,10)))
        return nodes

    def learn(self):
        pass





if __name__ == "__main__":
    net = NN()
    net.learn()
