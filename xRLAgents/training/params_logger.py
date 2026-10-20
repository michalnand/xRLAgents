import json
import os

def param_logger(config, result_path):
    if not os.path.exists(result_path):
        os.makedirs(result_path, exist_ok=True)

    result_file_name = os.path.join(result_path, "config_params.json")

    result = {}
    attrs = sorted(dir(config)) 

    for attr in attrs:
        # Skip dunder attributes
        if not attr.startswith("__"):
            try:
                value = getattr(config, attr)
                
                # Ensure it's not a method/callable AND check the value's type
                if not callable(value) and isinstance(value, (int, float, str, bool)):
                    result[attr] = value
            except Exception:
                # Catch properties or attributes that might raise errors on access
                pass

    result["hyperparameters"] = config.hyperparameters

    with open(result_file_name, "w") as f:
        json.dump(result, f, indent=4)