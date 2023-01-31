import math

class Node():
    def __init__(self,bias,index,links):
        self.index=index
        self.bias=bias
        self.links=links
        self.values=[]
    
    def calcSigmoid(self,x):
        # calculate the sigmoid
        return 1/(1+pow(math.e,-x))
        
    def test(self,value):
        self.values.append(value)

    def propagate(self):
        # gives values to the next layer
        value = self.calcValue()
        if self.index[0]==3:
            print(self.index[1],': ',value)
            return
        for i in range(len(self.nextL)):
            self.nextL[i].test(value*self.links[self.index[0]][self.index[1]][i])

    def calcValue(self):
        # calculate value
        return self.calcSigmoid(sum(self.values)+self.bias)

    def setLayers(self,layers):
        self.layers=layers
        if self.index[0] != 0:
            self.prevL=self.layers[self.index[0]-1]
        if self.index[0] != 3:
            self.nextL=self.layers[self.index[0]+1]
