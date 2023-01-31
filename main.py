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
        self.layers.append(self.generateNodes(784,0))
        self.layers.append(self.generateNodes(16,1))
        self.layers.append(self.generateNodes(16,2))
        self.layers.append(self.generateNodes(10,3))
        self.deliverLayers()
        self.data = openData()
        self.learn(next(self.data))

    def generateNodes(self,count,index):
        # generate a list of nodes of length <count> with index <index>
        nodes = []
        for a in range(count):
            nodes.append(Node(bias=random.randint((-10),10),index=(index,a),links=self.links))
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

    def analyze(self,image):
        i=0
        for row in image:
            print(row)
            for data in row:
                self.layers[0][i].analyze(data)
                i+=1
        for layer in self.layers:
            results = []
            for node in layer:
                results.append(node.propagate())
        print(results)
        return results

    def deliverLayers(self):
        for layer in self.layers:
            for node in layer:
                node.setLayers(self.layers)

    def learn(self, dataset):
        label, image = dataset
        res= self.analyze(image)
        wres=self.getWantedResults(label)
        self.cost=self.getCost(res,wres)
        print(self.cost)

    def getWantedResults(self,label):
        wanted_results=[]
        for i in range(10):
            wanted_results.append(0.0)
        wanted_results[label-1]=1.0
        return wanted_results

    def getCost(self,res,wres):
        cost = 0.0
        for i in range(len(res)):
            cost += pow(res[i]-wres[i],2)
        return cost


if __name__ == "__main__":
    net = NN()
