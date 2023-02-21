class TextPart:
    def __init__(self,text,predictions):
        self.text=text
        self.predictions=predictions

    def __str__(self):
        return f'{self.text} : {self.predictions}'

    def is_only_text(self):
        return (self.predictions==None)