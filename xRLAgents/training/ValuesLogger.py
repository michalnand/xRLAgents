

class ValuesLogger:
    def __init__(self, logger_name = "values_logger", add_to_summary = True):
        self.logger_name = logger_name
        self.values = {}

        self._add_to_summary = add_to_summary

    def add_to_summary(self):
        return self._add_to_summary

    def add(self, name, value, alpha_smoothing = 1.0, dp = 5):
        tmp = float(value)
        if name in self.values:
            self.values[name] = round((1.0 - alpha_smoothing)*self.values[name] + alpha_smoothing*tmp, dp)
        else:
            self.values[name] = round(tmp, dp)

    def add_dictionary(self, dictionary):
        for key in dictionary:
            self.add(str(key), dictionary[key])

    def get_str(self, decimals = 5): 
        result = "" 

        for index, (key, value) in enumerate(self.values.items()):
            try:
                s = str(round(value, decimals)) + ", "
            except:
                print("error conversion ", key, "\n", value)
                raise TypeError("Only floats are allowed")

            result+= s

        result = result[:-2]

        return result 
    
   
    
    def get_name(self):
        return self.logger_name
    
