import torch

class PPOAgentBase:
    def __init__(self):
        self.model = None

    def print_params(self):
        print("\n\n")
        attrs = sorted(dir(self)) 
        
        for attr in attrs:
            # Skip dunder attributes
            if not attr.startswith("__"):
                try:

                    value = getattr(self, attr)
                    if not callable(value) or attr == "model":
                        res = str(value)
                        print(attr, "=", res)
                    
                except Exception:
                    pass
            
        print("\n\n")

    def save(self, result_path):
        torch.save(self.model.state_dict(), result_path + "/checkpoints/model_weights.pt")
        torch.save(self.model, result_path + "/checkpoints/model.pt")

    def load(self, result_path):
        self.model.load_state_dict(torch.load(result_path + "/checkpoints/model_weights.pt", weights_only=True))
