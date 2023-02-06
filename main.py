#!/usr/bin/env python
import random, json
from node import Node
from dataImport import *
# Main class
class NN:
    def __init__(self):
        # generate links with random weigths
        #self.links=[
        #    self.generateLinks(784,16),
        #    self.generateLinks(16,16),
        #    self.generateLinks(16,10)]
        #self.biases=self.generateBias((784,16,16,10))
        self.load()
        # generate nodes
        self.layers=[]
        self.layers.append(self.generateNodes(784,0))
        self.layers.append(self.generateNodes(16,1))
        self.layers.append(self.generateNodes(16,2))
        self.layers.append(self.generateNodes(10,3))
        self.deliverLayers()
        self.data = openData()
        self.learn(next(self.data))
        self.save()

    def generateNodes(self,count,index):
        # generate a list of nodes of length <count> with index <index>
        nodes = []
        for a in range(count):
            nodes.append(Node(biases=self.biases,index=(index,a),links=self.links))
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

    def generateBias(self,layers):
        # same as generateNodes but for biases
        biases = []
        for layer in layers:
            col = []
            for i in range(layer):
                col.append(random.randint((-10),10))
            biases.append(col)
        return biases

    def analyze(self,image):
        # analyse an image and return activations of the last layer
        i=0
        for row in image:
            for data in row:
                self.layers[0][i].analyze(data)
                i+=1
        for layer in self.layers:
            results = []
            for node in layer:
                results.append(node.propagate())
        return results

    def deliverLayers(self):
        # give all Nodes a list with all the Layers
        for layer in self.layers:
            for node in layer:
                node.setLayers(self.layers)

    def learn(self, dataset):
        # analyze an image and learn
        label, image = dataset
        res= self.analyze(image)
        wres=self.getWantedResults(label)
        cost=self.getCost(res,wres)
        print(self.highestResult(res))
        print(cost)
        for layer in list(reversed(self.layers)):
            for node in layer:
                node.learn(cost)

    def highestResult(self,results:list):
        # return the index of the highest number in <results>
        a = list(results)
        a.sort()
        return results.index(a[len(a)-1])

    def getWantedResults(self,label):
        # returns a list with everything being 0 except the label of the image
        wanted_results=[]
        for i in range(10):
            wanted_results.append(0.0)
        wanted_results[label-1]=1.0
        return wanted_results

    def getCost(self,res,wres):
        # calculate the cost of the network
        cost = 0.0
        for i in range(len(res)):
            cost += pow(res[i]-wres[i],2)
        return cost
    
    def save(self):
        # saves weigths and biases to files
        with open('weigths.json','w') as file:
            json.dump(self.links,file)
        with open('biases.json','w') as file:
            json.dump(self.biases,file)

    def load(self):
        # loads weigths and biases from files
        with open('weigths.json','r') as file:
            self.links = json.load(file)
        with open('biases.json','r') as file:
            self.biases = json.load(file)


if __name__ == "__main__":
    net = NN()
