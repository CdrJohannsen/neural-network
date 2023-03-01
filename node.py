import math

class Node():
    biases=[]
    links=[]
    def __init__(self,biases,index,links):
        self.index=index
        self.biases=biases
        self.links=links
        self.values=[]
        self.lf=1
        self.w_value=0.5
    
    def calcSigmoid(self,x):
        # calculate the sigmoid
        return 1/(1+pow(math.e,-x))
        
    def analyze(self,value):
        self.values.append(value)

    def propagate(self):
        # gives values to the next layer
        self.values_bu=list(self.values)
        self.value = self.calcValue()
        self.values=[]
        if self.index[0]==3:
            return self.value
        for i in range(len(self.nextL)):
            self.nextL[i].analyze(self.value*self.links[self.index[0]][self.index[1]][i])

    def calcValue(self):
        # calculate value
        self.raw_value=0
        self.raw_value= sum(self.values_bu)+self.biases[self.index[0]][self.index[1]]
        return self.calcSigmoid(self.raw_value)

    def setLayers(self,layers):
        self.layers=layers
        if self.index[0] != 0:
            self.prevL=self.layers[self.index[0]-1]
        self.nextL=self.layers[self.index[0]+1]

    def learn(self,cost,links,biases):
        self.links=links
        self.biases=biases
        changes=[]
        for node in self.nextL:
            changes.append(self.calcChanges(node))
            #print(changes[-1])
        self.baseChange=sum(i[0] for i in changes)
        #self.biases[self.index[0]][self.index[1]]-=self.lf*self.baseChange
        if self.index[0]==3:
            return (self.links, self.biases)
        for i in range(len(self.nextL)-1):
            #print(self.lf*self.baseChange*changes[i-1][1])
            self.links[self.index[0]][self.index[1]][i-1]-=self.lf*self.baseChange*changes[i-1][1]
        # change bias and weight
        self.w_value=self.calcValue()
        self.values=[]
        return (self.links, self.biases)
        

    def calcChanges(self,nextNode):
        cost_value = 2*(nextNode.w_value-nextNode.value)
        value_rawValue = self.derivSigmoid(nextNode.raw_value)
        rawValue_weigth = self.raw_value
        #print(cost_value,value_rawValue,rawValue_weigth)
        change = cost_value*value_rawValue
        return change, rawValue_weigth

    def derivSigmoid(self, raw_value):
        sigmoid = self.calcSigmoid(raw_value)
        return sigmoid*(1-sigmoid)
