from transformers import pipeline

from src.strings import StringUtils
import pandas as pd
from tabulate import tabulate
mock=[
            {'text' : 'בראשית ברא אלוהים את ','predictions':None},
            {'text':'?','predictions':[{'value':'השמיים','p':0.8},{'value':'הידיים','p':0.1}]},
            {'text': ' ואת הא', 'predictions': None},
            {'text': '?', 'predictions': [{'value': 'ר', 'p': 0.86}, {'value': 'מ', 'p': 0.69}]},
            {'text': 'ץ', 'predictions': None},
        ]

class Model():

    def __init__(self,model_path):
        self.model_path=model_path

    def calc(self,text,min_p=0.01):

        mlm=pipeline("fill-mask", model=self.model_path)
        preds=self._predict_all(mlm, text)
        splited_text=text.split('?')
        res=[]
        for i,part in enumerate(splited_text):
            res.append({'text': part, 'predictions': None})
            if i==len(splited_text)-1:
                break
            next_preds={'text': '?', 'predictions': []}
            for idx,pred in preds.iterrows():
                if pred['token_index']==i:
                    if pred['score']>min_p:
                        next_preds['predictions'].append({'value': pred['token'], 'p': round(pred['score'], 3)})
            res.append(next_preds)
        return res

    def _predict_all(self,mlm, txt):
        count = txt.count('?')
        input_text = StringUtils.insert_masks(txt)
        all_preds = []
        for i in range(count):
            preds = mlm(input_text)
            try:
                input_text = input_text.replace('[MASK]', preds[0][0]['token_str'], 1)
                for pred in preds[0]:
                    all_preds.append({'token_index':i,'token':pred['token_str'],'score':pred['score']})
            except:
                input_text = input_text.replace('[MASK]', preds[0]['token_str'], 1)
                pred=preds[0]
                all_preds.append({'token_index':i,'token':pred['token_str'],'score':pred['score']})
        return  pd.DataFrame(all_preds)
