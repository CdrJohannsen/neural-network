import math

class Node():
    def __init__(self,bias,index,links,layers):
        self.index=index
        self.bias=bias
        self.links=links
        self.layers=layers
        self.values=[]
        if self.index[0] != 0:
            self.prevL=layers[self.index[0]-1]
        if self.index[0] != 3:
            self.nextL=layers[self.index[0]+1]
    
    def calcSigmoid(self,x):
        # calculate the sigmoid
        return 1/(1+pow(math.e,-x)
        
    def test(self,value):
        self.values.append(value)

    def propagate(self):
        # gives values to the next layer
        value = self.calcValue()
        if self.index[0]==3:
            print(self.index[1],': ',value)
        for i in range(len(self.nextL)):
            self.nextL[i].test(value*self.links[self.index[0]][self.index[1]][i])

    def calcValue(self):
        # calculate value
        return self.calcSigmoid(sum(self.values)+bias)
