class DNode():
    def __init__(self):
        self.value=0.0
        self.w_value=self.value
        self.baseChange=0.0
        
    def analyze(self,value):
        pass

    def propagate(self):
        pass

    def setLayers(self,layers):
        pass

    def learn(self,cost,links,biases):
        return (links,biases)

    def setValue(self,value):
        self.value=value
        self.w_value=value
        #self.baseChange=value
