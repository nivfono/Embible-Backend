from abc import ABC
from typing import List

from transformers import AutoModelForMaskedLM, AutoTokenizer

from src.classes.text_part import TextPart


class Model(ABC):
    def __init__(self,model_path):
        self.model_path=model_path
        self.model=AutoModelForMaskedLM.from_pretrained(model_path)
        self.tokenizer=AutoTokenizer.from_pretrained(model_path)

    def predict(self,text:str,min_p:float)->List[TextPart]:
        pass