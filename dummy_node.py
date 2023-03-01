class DNode():
    def __init__(self):
        self.value=0.0
        self.baseChange=0.0
        
    def analyze(self,value):
        pass

    def propagate(self):
        pass

    def setLayers(self,layers):
        pass

    def learn(self,cost):
        return ([],[])

    def setValue(self,value):
        self.value=value
        #self.baseChange=value
