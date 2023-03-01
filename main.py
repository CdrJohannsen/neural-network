#!/usr/bin/env python
import random, json, sys
from node import Node
from dummy_node import DNode
from dataImport import *
# Main class
class NN:
    def __init__(self):
        # generate links with random weigths
        if len(sys.argv)>=2 and sys.argv[1]=='--reset':
            self.links=[
                self.generateLinks(784,16),
                self.generateLinks(16,16),
                self.generateLinks(16,10)]
            self.biases=self.generateBias((784,16,16,10))
        else:
            self.load()
        # generate nodes
        self.layers=[]
        self.layers.append(self.generateNodes(784,0))
        self.layers.append(self.generateNodes(16,1))
        self.layers.append(self.generateNodes(16,2))
        self.layers.append(self.generateNodes(10,3))
        self.layers.append([])
        self.deliverLayers()
        self.data = openData()
        d=next(self.data)
        for asd in range(10):
            self.learn(d)
            self.save()
            self.load()
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
                row.append(random.random()*random.choice((-1,1)))
            col.append(row)
        return col

    def generateBias(self,layers):
        # same as generateNodes but for biases
        biases = []
        for layer in layers:
            col = []
            for i in range(layer):
                col.append(random.randint((-1),1))
            biases.append(col)
        return biases

    def analyze(self,image):
        # analyse an image and return activations of the last layer
        i=0
        for row in image:
            for data in row:
                self.layers[0][i].analyze(data/255)
                i+=1
        for layer in self.layers[:len(self.layers)-1]:
            results = []
            for node in layer:
                results.append(node.propagate())
        print(results)
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
        self.layers[4]=list(wres)
        self.deliverLayers()
        cost=self.getCost(res,wres)
        print(f'Wanted: {label}')
        print(f'Result: {self.highestResult(res)}')
        print(f'Cost: {cost}')
        for layer in list(reversed(self.layers)):
            for node in layer:
                (self.links, self.biases)=node.learn(cost)

    def highestResult(self,results:list):
        # return the index of the highest number in <results>
        a = list(results)
        a.sort()
        return results.index(a[-1])

    def getWantedResults(self,label):
        # returns a list with everything being 0 except the label of the image
        wanted_results=[]
        for i in range(10):
            wanted_results.append(DNode())
        wanted_results[label-1].setValue(1.0)
        return wanted_results

    def getCost(self,res,wres):
        # calculate the cost of the network
        cost = 0.0
        for i in range(len(res)):
            cost += pow(res[i]-wres[i].value,2)
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
