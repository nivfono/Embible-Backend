import copy
import re
from abc import ABC, abstractmethod
from itertools import groupby

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForMaskedLM

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
        self.model=AutoModelForMaskedLM.from_pretrained(model_path)
        self.tokenizer=AutoTokenizer.from_pretrained(model_path)

    def predict(self,text,min_p=0.01):
        if '?' not in text:
            return [{'text' : text,'predictions':None}]
        # predict
        preds = self.get_predictions(text)

        # split to parts so we can turn these parts to list of {'text': 'part', 'predictions': [...]}
        splited_text=re.split('(\?)',text)
        splited_text=[i for i in splited_text if i != '']
        tmp=[]

        for i,t in enumerate(splited_text):
            if splited_text[i-1]=='?' and t=='?':
                continue
            tmp.append(t)


        splited_text=tmp
        # parsing the parts to {'text': 'part', 'predictions': [...]}
        res=[]
        pred_index=0
        for i,part in enumerate(splited_text):
            next_pred={'text': part, 'predictions': None}
            if '?' in part:
                next_pred['predictions'] = preds[pred_index]
                pred_index+=1
            res.append(next_pred)
        return res

    def get_predictions(self,text):
        text=StringUtils.insert_masks(text)
        # tokenize input
        token_ids = self.tokenizer.encode(text, return_tensors='pt')
        # masked token indexes
        masked_pos = [mask.item() for mask in (token_ids.squeeze() == self.tokenizer.mask_token_id).nonzero()]
        # predict
        with torch.no_grad():
            output = self.model(token_ids)
        last_hidden_state = output[0].squeeze()
        # extract predictions from last_hidden_state
        res = []
        for mask_index in masked_pos:
            mask_hidden_state = last_hidden_state[mask_index]
            topk_predictions=torch.topk(mask_hidden_state, k=10, dim=0)
            topk_preds_idx = topk_predictions[1]
            predictions = [self.tokenizer.decode(i.item()).strip() for i in topk_preds_idx]
            predictions=[i.replace('##', '') for i in predictions]
            probs = topk_predictions[0].softmax(-1)
            # parse the data into interface
            cur_pred_results = []
            for pred_index, pred in enumerate(predictions):
                cur_pred_results.append({'value': pred, 'p': round(probs[pred_index].item(), 2)})
            res.append(cur_pred_results)

        return res



    # def calc(self,text,min_p=0.01):
    #     mlm=pipeline("fill-mask", model=self.model_path)
    #     preds=self._predict_all(mlm, text)
    #     splited_text=text.split('?')
    #     res=[]
    #     for i,part in enumerate(splited_text):
    #         res.append({'text': part, 'predictions': None})
    #         if i==len(splited_text)-1:
    #             break
    #         next_preds={'text': '?', 'predictions': []}
    #         for idx,pred in preds.iterrows():
    #             if pred['token_index']==i:
    #                 if pred['score']>min_p:
    #                     next_preds['predictions'].append({'value': pred['token'], 'p': round(pred['score'], 3)})
    #         res.append(next_preds)
    #     return res
    #
    # def _predict_all(self,mlm, txt):
    #     count = txt.count('?')
    #     input_text = StringUtils.insert_masks(txt)
    #     all_preds = []
    #     for i in range(count):
    #         preds = mlm(input_text)
    #         try:
    #             input_text = input_text.replace('[MASK]', preds[0][0]['token_str'], 1)
    #             for pred in preds[0]:
    #                 all_preds.append({'token_index':i,'token':pred['token_str'],'score':pred['score']})
    #         except:
    #             input_text = input_text.replace('[MASK]', preds[0]['token_str'], 1)
    #             pred=preds[0]
    #             all_preds.append({'token_index':i,'token':pred['token_str'],'score':pred['score']})
    #     return  pd.DataFrame(all_preds)
