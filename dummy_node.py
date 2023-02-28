class DNode():
    def __init__(self):
        self.value=0.0
        self.baseChange=0.0
        
    def analyze(self,value):
        pass

    def propagate(self):
        pass

    def calcValue(self):
        # calculate value
        self.raw_value= sum(self.values)+self.biases[self.index[0]][self.index[1]]
        return self.calcSigmoid(self.raw_value)

    def setLayers(self,layers):
        pass

    def learn(self,cost):
        pass

    def setValue(self,value):
        self.value=value
        self.baseChange=value
