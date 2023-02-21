import config
from src.classes.prediction import Prediction
from src.classes.text_part import TextPart
from src.classes.ensemble_result import EnsembleResult
from src.model.sequential_model import SequentialModel
from src.model.standard_model import StandardModel


class Ensamble():

    def __init__(self):
        self.word_model=StandardModel(config.configs['word_model_path'])
        self.subword_model=StandardModel(config.configs['subword_model_path'])
        self.char_model=StandardModel(config.configs['char_model_path'])
        self.sequential_char_model = SequentialModel(config.configs['char_model_path'])


    def predict(self,text,min_p=0.1):
        self.last_word_model_preds=self.word_model.predict(text,min_p)
        self.last_char_model_sequential_preds = self.sequential_char_model.predict(text,min_p)
        self.last_subword_model_preds =  self.subword_model.predict(text,min_p)
        self.last_char_model_preds = self.char_model.predict(text,min_p)

        res=[]
        for pred_index in range(len(self.last_word_model_preds)):
                if pred_index>=len(self.last_word_model_preds):
                    break
                if self.last_word_model_preds[pred_index].is_only_text():
                    res.append(self.last_word_model_preds[pred_index])
                else:
                    res.append(self._get_pred(pred_index))

        return EnsembleResult(list(filter(lambda x:x!=None,res)))


    def _get_pred(self,pred_index):
        if(len(self.last_word_model_preds)<=pred_index):
            return
        # handle extreme points
        if '?' not in self.last_word_model_preds[pred_index].text:
            return self.last_word_model_preds[pred_index]

        before,after=self._get_before_after(pred_index)
        return self._get_pred_by_type( before, after, pred_index)

    def _get_before_after(self,pred_index):
        if pred_index ==0:
            before=TextPart(' ',None)
        else:
            before=self.last_word_model_preds[pred_index-1]
        if pred_index ==len(self.last_word_model_preds)-1:
            after = TextPart(' ',None)
        else:
            after = self.last_word_model_preds[pred_index + 1]

        return before,after

    def _get_pred_by_type(self,before,after,pred_index):
        if self._is_index_starts_a_word(pred_index):
            return self._get_word_prediction_at_index(pred_index)
        elif after.text[0]!=' ' and before.text[-1]!=' ':
            return self._get_char_prediction_at_index(pred_index)
        else:
            return self._get_subword_prediction_at_index(pred_index)

    def _get_word_prediction_at_index(self,index):
        preds=self.last_word_model_preds[index].predictions
        res_preds=[]
        word_len=self._get_this_word_length(index,self.last_word_model_preds)
        # self._trim_folowing_q_marks(index,word_len)
        for pred in preds:
            if(len(pred.value)==word_len):
                res_preds.append(pred)
        if len(res_preds)==0:
            res_preds=self._get_sequentially_pred_at_index(index,word_len)

        return TextPart('?',res_preds)
    def _get_subword_prediction_at_index(self,index):
        return self.last_subword_model_preds[index]
    def _get_char_prediction_at_index(self,index):
        return self.last_char_model_preds[index]

    def _get_this_word_length(self,index,preds):
        count=0
        for i in range(index,len(preds)):
            if(preds[i].text=='?'):
                count+=1;
            else:
                return count
        return count

    def _get_sequentially_pred_at_index(self,index,word_len):
        preds=[pred.predictions[0] for pred in self.last_char_model_sequential_preds[index:index+word_len] 
               if pred.predictions !=None and pred.predictions!=[] ]
        if preds==[]:
            return []
        res_txt=''
        pred_score_sum=0
        for pred in preds:
            res_txt+=pred.value
            pred_score_sum+=pred.score
        avg_score=pred_score_sum/len(preds)
        return [Prediction(res_txt,round(avg_score,3))]

    def _is_index_starts_a_word(self,pred_index):
        for i in range(pred_index,len(self.last_word_model_preds)):
            if self.last_word_model_preds[i].text==' ':
                break
            if self.last_word_model_preds[i].text!='?':
                return False
        return True
    def _trim_folowing_q_marks(self,index,word_len):
        for i in range(index,index+word_len):
            self.last_word_model_preds.pop(index)