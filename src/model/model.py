from abc import ABC

from transformers import AutoModelForMaskedLM, AutoTokenizer


class Model(ABC):
    def __init__(self,model_path):
        self.model_path=model_path
        self.model=AutoModelForMaskedLM.from_pretrained(model_path)
        self.tokenizer=AutoTokenizer.from_pretrained(model_path)

    def predict(self,text,min_p):
        pass