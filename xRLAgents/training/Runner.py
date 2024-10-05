import multiprocessing
import time
import torch

import importlib.util
import sys
import os

class Runner:

    # devices = [cuda:0, cuda:1 ... ]
    def __init__(self, experiments, devices, delay_s = 5.0):
        multiprocessing.set_start_method('spawn')
        #multiprocessing.set_start_method('fork')

        process = []
        for i in range(len(experiments)):

            device = devices[i%len(devices)]

            #with multiprocessing.Pool(1) as p:
            #    p.map(module.run, [])

            params = [experiments[i], i, device]
            p = multiprocessing.Process(target=self._run, args=params)
            process.append(p)

        for p in process:
            p.start()
            time.sleep(delay_s)

        for p in process:
            p.join()

        for p in process:
            del p

        print("Runner : experiments done")

    '''
    def _run(self, experiment, i, device):
        experiment = experiment + ".main"
        print("Runner : starting ", i, experiment)

        try:
            if "cuda" in device:
                torch.cuda.set_device(device)
                print("Runner : device   ", device)
        except:
            pass

        module = __import__(experiment)
        module.run()

        print("Runner : ending ", i, experiment)
    '''


    


    def _run(self, experiment_path, i, device):
        print("Runner : starting ", i, experiment_path)

        try:
            if "cuda" in device:
                torch.cuda.set_device(device)
                print("Runner : setting device   ", device)
        except:
            pass

        # Change working directory to the experiment path
        os.chdir(experiment_path)
        
        # Temporarily add the experiment directory to sys.path to ensure imports work
        sys.path.insert(0, experiment_path)
        
        # Import the main module dynamically and run it
        module_name = 'main'
        module = __import__(module_name)
        
        module.run()

        print("Runner : ending ", i, experiment_path)

