import re

class Remove:
    def __init__(self, data, field_name, remove_english=True, remove_invalid=True, remove_company=True, verbose=True):
        self.verbose = verbose

        self.data = data
        self.field_name = field_name
        if remove_english:
            self.remove_english_models()
            self.log(
                f'Number of rows after removing English instance: {self.data.shape[0]}')
        if remove_invalid:
            self.remove_invalid_models()
            self.log(
                f'Number of rows after removing invalid instance: {self.data.shape[0]}')
        if remove_company:
            self.remove_company_ads()
            self.log(
                f'Number of rows after removing company ads: {self.data.shape[0]}')

    def log(self, text):
        if self.verbose:
            print(text)

    @staticmethod
    def is_ascii(text):
        return all(ord(char) <= 127 for char in text)

    def remove_english_models(self):
        self.data = self.data[~self.data[self.field_name].apply(self.is_ascii)]

    @staticmethod
    def is_valid_model(text):
        match = re.findall(r'مدل\s*(\d{2,4})|مدل\_*(\d{2,4})|مدل\-*(\d{2,4})', text)
        if len(match) > 1:
            return False
        # Use a set for faster membership checks
        keywords = {'تعویض', 'معاوضه', 'معامله', 'معاوض'}
        return not any(keyword in text for keyword in keywords)

    def remove_invalid_models(self):
        self.data = self.data[self.data[self.field_name].apply(
            self.is_valid_model)]

    def remove_company_ads(self):
        self.data = self.data[~self.data[self.field_name].str.contains(
            'اقساط')]