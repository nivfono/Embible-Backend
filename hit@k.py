
import json
import time

import pandas as pd
import config
from src.ensamble import Ensamble
from huggingface_hub import login

login(config.configs['hf_token'])
ens=Ensamble()
with open('test.json','r') as r:
    test_data=json.load(r)

for entry in test_data:
    print(entry)
    res=ens.predict(entry['text']).get_only_k_predictions(1)
    print(res.get_ui_format())
    break