

class ValuesLogger:
    def __init__(self, logger_name = "values_logger"):
        self.logger_name = logger_name
        self.values = {}

    def add(self, name, value, smoothing = 0.1):
        if name in self.values:
            self.values[name] = (1.0 - smoothing)*self.values[name] + smoothing*value
        else: 
            self.values[name] = value

    def get_str(self, decimals = 5): 
        result = "" 

        for index, (key, value) in enumerate(self.values.items()):
            s = str(round(value, decimals)) + ", "
            result+= s

        result = result[:-2]

        return result 
    
    def get_named_str(self, decimals = 5): 
        result = "{\"" + self.logger_name + "\" : ["

        result+= self.get_str(decimals)
        result+= "]}   "

        return result 
    
    def get_values(self, decimals = 5):
        result = []
        for index, (key, value) in enumerate(self.values.items()):
            result.append(round(value, decimals))

        return result
    
    def get_name(self):
        return self.logger_name
    

'''
class ValuesLogger:
    def __init__(self, file_name):
        self.values = {}
        self.f = open(file_name, "w")

    def add(self, name, value, smoothing = 0.1):
        if name in self.values:
            self.values[name] = (1.0 - smoothing)*self.values[name] + smoothing*value
        else: 
            self.values[name] = value

    def save(self, decimals = 5):
        s = self.get_str(decimals)

        self.f.write(s + "\n")
        self.f.flush()

    def print(self, decimals = 3):
        s = self.get_str(decimals)
        print(s)

    def close(self):
        self.f.flush()
        self.f.close()

    def get_str(self, decimals = 6): 
        result = "" 

        for index, (key, value) in enumerate(self.values.items()):
            s = str(round(value, decimals)) + " "
            result+= s

        return result 
    
    def get_values(self, decimals = 6):
        result = []
        for index, (key, value) in enumerate(self.values.items()):
            result.append(round(value, decimals))

        return result
'''