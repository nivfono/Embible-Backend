class Prediction:
    def __init__(self,value,score):
        self.value=value
        self.score=score

    def __str__(self):
        return f'({self.value},{self.score})'