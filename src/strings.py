class StringUtils():

    @staticmethod
    def  insert_masks(text):
        res = ''
        for letter in text:
            add = ''
            if letter == '?':
                add = '[MASK]'
            else:
                add = letter
            res += add
        return res

