import config
from src.model.model import Model


class Ensamble():

    def __init__(self):
        self.word_model=Model(config.configs['word_model_path'])
        self.subword_model=Model(config.configs['subword_model_path'])
        self.char_model=Model(config.configs['char_model_path'])

    def predict(self,text):
        self.last_word_model_preds=self.word_model.predict(text)
        self.last_subword_model_preds = self.subword_model.predict(text)
        self.last_char_model_preds = self.char_model.predict(text)

        res=[]
        for pred_index in range(len(self.word_model.predict(text))):
            res.append(self._get_pred_by_type(pred_index))
        return res


    def _get_pred_by_type(self,pred_index):
        # handle extreme points
        if '?' not in self.last_word_model_preds[pred_index]['text']:
            return self.last_word_model_preds[pred_index]
        if pred_index ==0:
            before={'text': ' '}
        else:
            before=self.last_word_model_preds[pred_index-1]
        if pred_index ==len(self.last_word_model_preds)-1:
            after = {'text': ' '}
        else:
            after = self.last_word_model_preds[pred_index + 1]
        #get type
        if after['text'][0]==' ' and before['text'][-1]==' ':
            print('word preds:',self.last_word_model_preds[pred_index]['predictions'])
            return self.last_word_model_preds[pred_index]
        elif after['text'][0]!=' ' and before['text'][-1]!=' ':
            print('char preds:', self.last_char_model_preds[pred_index]['predictions'])
            return self.last_char_model_preds[pred_index]
        else:
            print('subword preds:', self.last_subword_model_preds[pred_index]['predictions'])
            return self.last_subword_model_preds[pred_index]
