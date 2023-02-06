import math

class Node():
    def __init__(self,biases,index,links):
        self.index=index
        self.biases=biases
        self.links=links
        self.values=[]
    
    def calcSigmoid(self,x):
        # calculate the sigmoid
        return 1/(1+pow(math.e,-x))
        
    def analyze(self,value):
        self.values.append(value)

    def propagate(self):
        # gives values to the next layer
        value = self.calcValue()
        if self.index[0]==3:
            return value
        for i in range(len(self.nextL)):
            self.nextL[i].analyze(value*self.links[self.index[0]][self.index[1]][i])

    def calcValue(self):
        # calculate value
        return self.calcSigmoid(sum(self.values)+self.biases[self.index[0]][self.index[1]])

    def setLayers(self,layers):
        self.layers=layers
        if self.index[0] != 0:
            self.prevL=self.layers[self.index[0]-1]
        if self.index[0] != 3:
            self.nextL=self.layers[self.index[0]+1]

    def learn(self,cost):
        changes=[]
        for i in range(len(self.nextL)):
            canges.append(self.calcChanges(self.nextL[i]))
        self.baseChange=sum(changes)/len(changes)
        # change bias and weight
        

    def calcChanges(self,nextNode):
        pass
