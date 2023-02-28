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
        self.value = self.calcValue()
        if self.index[0]==3:
            return self.value
        for i in range(len(self.nextL)):
            self.nextL[i].analyze(self.value*self.links[self.index[0]][self.index[1]][i])

    def calcValue(self):
        # calculate value
        self.raw_value= sum(self.values)+self.biases[self.index[0]][self.index[1]]
        return self.calcSigmoid(self.raw_value)

    def setLayers(self,layers):
        self.layers=layers
        if self.index[0] != 0:
            self.prevL=self.layers[self.index[0]-1]
        self.nextL=self.layers[self.index[0]+1]

    def learn(self,cost):
        changes=[]
        for i in range(len(self.nextL)-1):
            changes.append(self.calcChanges(self.nextL[i]))
        self.baseChange=sum(i[0] for i in changes)/len(changes)
        print(self.baseChange)
        # change bias and weight
        

    def calcChanges(self,nextNode):
        cost_value = 2*(self.value-nextNode.value)
        value_rawValue = self.derivSigmoid(self.raw_value)
        rawValue_weigth = sum(self.values)
        #print(cost_value,value_rawValue,rawValue_weigth)
        change = cost_value*value_rawValue
        return change, rawValue_weigth

    def derivSigmoid(self, raw_value):
        sigmoid = self.calcSigmoid(raw_value)
        return sigmoid*(1-sigmoid)
