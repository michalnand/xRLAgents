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


    


    def _run(self, experiment, i, device):
        print("Runner : starting ", i, experiment)

        module_dir = os.path.dirname(experiment)

        if module_dir not in sys.path:
            sys.path.insert(0, module_dir)


        try:
            if "cuda" in device:
                torch.cuda.set_device(device)
                print("Runner : device   ", device)
        except:
            pass

        module = __import__(experiment + ".main")
        module.run()

        print("Runner : ending ", i, experiment)

