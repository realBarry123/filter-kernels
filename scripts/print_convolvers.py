from dotenv import load_dotenv
load_dotenv()

import warnings
from filter_kernels import convolvers
from pathlib import Path

HERE = Path(__file__).resolve().parent

for model_name in convolvers.AVAILABLE_MODELS:
    try:
        model = convolvers.load(model_name)
    except NotImplementedError:
        warnings.warn(f"loading \"{model_name}\" not implemented")
        continue

    path = f"{HERE}/artifacts/str_{model_name}.txt"
    with open(path, "w+") as f: 
        f.write(model.__str__())
    print(f"saved printout to \"{path}\"")
