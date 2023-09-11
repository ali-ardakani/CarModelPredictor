import re
import pandas as pd
from hazm import Normalizer
from string import punctuation
from datetime import datetime
from jdatetime import datetime as jd
from tqdm import tqdm
import numpy as np

tqdm.pandas()

class Clean:

    def __init__(self, data, field_name) -> None:
        self.normalizer = Normalizer()
        self.data = data
        self.field_name = field_name
    
    @staticmethod
    def remove_punctuation(text):
        punctuation_ = punctuation+'،'

        return text.translate(str.maketrans(punctuation_, ' '*len(punctuation_)))   
    
    @staticmethod
    def replace_with_space(text):
        return text.replace('\u200c', ' ')
    
    def normalize(self, row):
        row[self.field_name] = self.normalizer.normalize(row[self.field_name])
        row[self.field_name] = self.remove_punctuation(row[self.field_name])
        row[self.field_name] = self.replace_with_space(row[self.field_name])
        return row
 
    def normalize_data(self):
        self.data = self.data.progress_apply(self.normalize, axis=1)
        return self.data
