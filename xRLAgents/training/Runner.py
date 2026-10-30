import multiprocessing
import time
import torch

class Runner:

    # devices = [cuda:0, cuda:1 ... ]
    def __init__(self, experiments_name, devices, delay_s = 5.0):
        multiprocessing.set_start_method('spawn')
        #multiprocessing.set_start_method('fork')

        process = []
        for i in range(len(experiments_name)):

            device = devices[i%len(devices)]

            #with multiprocessing.Pool(1) as p:
            #    p.map(module.run, [])

            params = [experiments_name[i], i, device]
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

    
    def _run(self, experiment, i, device):
        print("Runner : starting ", i, experiment, device)

        try:
            if "cuda" in device:
                torch.cuda.set_device(device)
                print("setting device to ", device)
        except:
            pass

        module = __import__(experiment)
        module.run()

        print("Runner : ending ", i, experiment)
    
