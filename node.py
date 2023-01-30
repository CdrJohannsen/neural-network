import math

class Node():
    def __init__(self,bias,index,links,layers):
        self.index=index
        self.bias=bias
        self.links=links
        self.layers=layers
        if self.index[0] != 0:
            self.prevL=layers[self.index[0]-1]
        if self.index[0] != 3:
            self.nextL=layers[self.index[0]+1]
    
    def calcSigmoid(self,x):
        return 1/(1+pow(math.e,-x)
        
    def test(self,value):
        for i in range(len(self.nextL)):
            self.nextL[i].test(value*self.links[self.index[0]][self.index[1]][i])
