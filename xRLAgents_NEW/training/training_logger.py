import json

import os

import numpy

class TrainingLogger:

    def __init__(self, result_path, log_name = "training"):

        self.result_curr = {}

        self.log_name = log_name

        self.result_file_name = result_path + log_name + ".jsonl"

        if not os.path.exists(result_path):
            os.makedirs(result_path)

        f = open(self.result_file_name, "w")
        f.close()

        print("creating logger ", self.result_file_name)

    def add(self, name, value, dp = None):
        if dp is not None:
            self.result_curr[name] = round(value, dp)
        else:
            self.result_curr[name] = value


    def add_average(self, name, values : list, dp = 6):
        values_mean = numpy.mean(values)
        values_std  = numpy.std(values)
        
        if dp is not None:
            self.result_curr[name + "_mean"] = round(values_mean, dp)
            self.result_curr[name + "_std"]  = round(values_std, dp)
        else:
            self.result_curr[name + "_mean"] = values_mean
            self.result_curr[name + "_std"]  = values_std

    def add_raw(self, dict_result):
        self.result_curr = dict(dict_result)


    def save(self, clear = False, verbose = False ):
        print("saving log ", self.result_file_name)
        f = open(self.result_file_name, "a+")
        str_result = json.dumps(self.result_curr)
        f.write(str_result + "\n")
        f.close()

        if clear:
            self.result_curr = {}
        
        if verbose:
            print(self.log_name, " : ")
            print(str_result)
            print("\n\n")

