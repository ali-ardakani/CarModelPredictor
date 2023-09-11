import re
import pandas as pd
from hazm import Normalizer
from string import punctuation
from datetime import datetime
from jdatetime import datetime as jd
from tqdm import tqdm
import numpy as np

tqdm.pandas()


class DataProcessor:
    def __init__(self, data, field_name='model', remove_english=True, remove_invalid=True, remove_company=True, verbose=True):
        self.verbose = verbose

        self.data = data
        self.field_name = field_name
        if remove_english:
            self.remove_english_models()
            self.log(
                f'Number of rows after removing English models: {self.data.shape[0]}')
        if remove_invalid:
            self.remove_invalid_models()
            self.log(
                f'Number of rows after removing invalid models: {self.data.shape[0]}')
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

    def print_data_summary(self):
        print(f'Number of rows after processing: {self.data.shape[0]}')


class DataNormalizer:
    STATES = ["اردبیل", "آذربایجان غربی", "آذربایجان شرقی", "بوشهر", "چهار محال و بختیاری", "فارس", "گیلان", "گلستان", "همدان", "هرمزگان", "ایلام", "اصفهان",
              "کرمان", "کرمانشاه", "خوزستان", "کهگیلویه و بویراحمد", "کردستان", "لرستان", "مرکزی", "مازندران", "قزوین", "قم", "سیستان و بلوچستان", "تهران", "یزد"]
    
    PATTERNS = [
        r"کارکرد\s*\d+",
        r'\d+\s*هزار\s*کارکرد',
        r'\d+\s*هزار\s*کیلومتر\s*کارکرد',
        r'\d+\s*هزار\s*کیلومتر',
        r'\d+\s*کیلومتر\s*کارکرد',
        r'\d+\s*کیلومتر',
        r'سیلندر\s*\d+'
    ]
    
    ADDITIONAL_INFORMATION = [
        "سند",
        "رنگ",
        "بیمه",
        "شرکتی",
        "تخفیف",
        "بدون",
        "دنده",
        "دست",
        "سوار",
        "تمیز",
        "تمام",
        "فول",
        "فنی",
        "دارد",
        "دارای",
        "داره",
        "دارن",
        "دارند",
        "دار",
        "بی",
        "آزاد",
        "ماشین",
        "کم",
        "کار",
        "شرایط",
        "ویژه",
        "موتور",
        "موتوری",
        "انژکتور",
        "انژکتوری",
        "در",
        "حد",
        "صفر",
        "صفرکیلومتر",
        "صفر کیلومتر",
        "کیلومتر",
        'لاکچری',
        'زیبا',
        'بسیار',
        "پلاک",
        "شده",
        "ملی",
        "گذر",
        "معاینه",
        "معاینه فنی",
        "تخفیف",
        "پای",
        "موقت",
        "موقتی",
        "خشک",
        "تحویل",
        "روز",
        "مونتاژ",
        "آپشنال",
        "فروشی",
        "فروش",
        "جدید",
        "نو",
        "خودرو",
        "ماشین",
        "متفاوت",
        "وارداتی",
        "و",
        "همان",
        "لحظه",
        "گارانتی",
        "فعال",
        "خیلی",
        "سالم",
        "کاملا",
        "کارکرد",
        "رخ",
        "آپشن",
        "با",
        "لوازم",
        "بیرنگ",
        "بیضربه",
        "سواری",
        "کارشناسی",
        "اپشنال",
        "درحد",
        "هیدرولیک",
        "ABS",
        "درحدصفر",
        "آماده",
        "مشابه",
        "لولزم",
        "ساده",
        "هزارتا",
        "بدنه",
        "اتولاتدار",
        "همراه",
        "نمایندگی",
        "معمولی",
        "دیلایت",
        "تصادفی",
        "واقعی",
        "عمومی",
        "وسلامت",
        "باربند",
        "مسقف",
        "عروسک",
        "کلاسیک",
        "پرند",
        "شاسی",
        "باربندچادر",
        "شدگی",
        "پانوراما",
        "سرامیک",
        "بدنه",
        "سقف",
        "وسقف",
        "فلز",
        "برقی",
        "فرمان",
        "وفلز",
        "صفرسقف",
        "هاچبک",
        "اول",
        "حدصفر",
        "دنده",
        "دنده‌ای",
        "دنده‌ ای",
        "برج",
        "دریچه",
        "سیمی",
        "نقدی",
        "گاز",
        "سوز",
        "روغن",
        "کارخانه",
        "انژکتر",
        "خوشگل",
        "بدونه",
        "فابریک",
        "وانژکتور",
        "پلمپ",
        "کمپانی",
        "فرمون",
        "ABSدار",
        "خطو",
        "خط",
        "خش",
        "ترمز",
        "ایینه",
        "وفنی",
        "کلکسیونی",
        "بشرط",
        "نقره",
        "متالیک",
        "کاربراتور",
        "تمیزسالم",
        "کامل",
        "بودن",
        "سرمه ای",
        "برگ",
        "یک",
        "خرجی",
        "هیچ",
        "انژکتورفنی",
        "سفارش",
        "آپکو",
        "زیاد",
        "خطوخش",
        "تصادف",
        "اولی",
        "سری",
        "وبدون",
        "خرج",
        "کار",
        "صفرنمایندگی",
        "امروز",
        "فقط",
        "مدیران",
        "درجا",
        "سرحال",
        "کروز",
        "کلاچ",
        "درکهریزک",
        "لکه",
        "به",
        "شهری",
        "نقطه",
        "بشرط",
        "ببمه",
        "ثالث",
        "یکسال",
        "فابریک",
        "خواب",
        "یک",
        "سال",
        "اسبی",
        "جلو",
        "عقب",
        "نیمه",
        "دوره",
        "خشگ",
        "باسیستم",
        "سرحال",
        "وتمیز",
        "شرط",
    ]
    
    PERSIAN_COLORS = [
            "سفید",      # White
            "مشکی",      # Black
            "قرمز",      # Red
            "آبی",       # Blue
            "سبز",       # Green
            "زرد",       # Yellow
            "نارنجی",    # Orange
            "بنفش",      # Purple
            "صورتی",     # Pink
            "قهوه‌ای",   # Brown
            "خاکستری",   # Gray
            "بژ",    # Beige
            "بنفش",      # Lavender
            "آبی کبود",  # Navy Blue
            "نیلی",      # Indigo
            "آبی آسمانی",  # Sky Blue
            "آبی توسکا",    # Turquoise
            "فیروزه‌ای",   # Teal
            "نقره‌ای",   # Silver
            "طلایی",       # Gold
            "نوک مدادی", "نوکمدادی",  # Pencil Lead
            "آلبالویی",  # Plum
            "گوجه",      # Tomato
        ]

    def __init__(self, data, field_name='model') -> None:
        self.normalizer = Normalizer()
        self.data = data
        self.field_name = field_name
        # self.locations = self.get_locations(data.location.unique())
        
        # Compile regex patterns once
        # self.locations = [re.escape(location) for location in self.locations]
        self.states = [re.escape(state) for state in self.STATES]
        self.PERSIAN_COLORS += [color + ' ای' for color in self.PERSIAN_COLORS]
        self.PERSIAN_COLORS += [color + 'ی' for color in self.PERSIAN_COLORS]
        self.PERSIAN_COLORS += [color + 'ای' for color in self.PERSIAN_COLORS]
        self.persian_colors = [re.escape(color) for color in self.PERSIAN_COLORS]
        self.patterns = [re.compile(pattern) for pattern in self.PATTERNS]
        self.additional_info = [re.escape(info) for info in self.ADDITIONAL_INFORMATION]


    def get_locations(self, locations):
        locations = self.data[self.data.location.notna(
        )].location.str.replace('\u200c', ' ')
        locations = locations.apply(
            lambda x: '' if re.findall(r'\d', x) else x)
        locations = locations.unique()
        locations = [self.normalizer.normalize(
            location) for location in locations]
        return locations

    @staticmethod
    def remove_cc(text):
        text = re.sub(r'(\d+)( cc| سی سی| سی‌سی| سیسی|cc|سی سی|سی‌سی|سیسی )','', text)
        text = re.sub(r'(cc |سی سی |سی‌سی |سیسی |cc|سی سی|سی‌سی|سیسی)(\d+)','', text)
        return text
    
    @staticmethod
    def remove_punctuation(text):
        punctuation_ = punctuation+'،'

        return text.translate(str.maketrans(punctuation_, ' '*len(punctuation_)))   
    
    def remove_color(self, text):

        for color in self.PERSIAN_COLORS:
            text = re.sub(rf'\b{re.escape(color)}\b', '', text)
        return text
    
    def remove_location(self, text):
        for location in self.locations + self.STATES:
            text = re.sub(rf'\b{re.escape(location)}\b', '', text)
        return text
    
    def remove_additional_information(self, text):
        for pattern in self.PATTERNS:
            text = re.sub(pattern, '', text)
        for info in self.ADDITIONAL_INFORMATION:
            text = re.sub(rf'\b{re.escape(info)}\b', '', text)
        return text
    
    def year(self, text):
        year = None
        matches = re.findall(r'مدل\s*(\d{2,4})|مدل\_*(\d{2,4})|مدل\-*(\d{2,4})', text)
        matches = set(match for match_group in matches for match in match_group if match)
        if len(matches) != 1:
            return None, None
        year = int(matches.pop()) if matches else None
        
        if year < 100:
            if year > 0 and year <= jd.now().year - 1400:
                year += 1400
            elif year >= 10 and year <= datetime.now().year+1 - 2000:
                year += 2000
            else:
                year += 1300
        if year < 1000:
            year += 1000
        if year >1300 and year <= jd.now().year +1:
            year = year + 621
        if year < 2000 or year > datetime.now().year + 1:
            year = None
        return year, re.search(r'مدل\s*(\d{2,4})', text)
    
    def remove_color_location_info(self, text):
        # Combine color and location patterns
        combined_pattern = '|'.join(self.persian_colors + self.locations + self.states + self.additional_info)
        combined_pattern = re.compile(combined_pattern)
        text = re.sub(combined_pattern, '', text)

        # Remove additional information
        for pattern in self.patterns:
            text = pattern.sub('', text)
        
        return text
    
    @staticmethod
    def replace_with_space(text):
        return text.replace('\u200c', ' ')
    
    @staticmethod
    def price(price):
        if pd.isna(price):
            return np.nan
        text = str(int(price))
        if len(set(text)) == 1:
            return None
        return price
    
    def normalize(self, row):
        try:
            row[self.field_name] = self.normalizer.normalize(row[self.field_name])
        except Exception as e:
            print(row[self.field_name])
            raise e
        # row[self.field_name] = self.remove_cc(row[self.field_name])
        row[self.field_name] = self.remove_punctuation(row[self.field_name])
        # row[self.field_name] = self.remove_color(row[self.field_name])
        # # row[self.field_name] = self.remove_location(row[self.field_name])
        # row[self.field_name] = self.remove_additional_information(row[self.field_name])
        # # row[self.field_name] = self.remove_color_location_info(row[self.field_name])
        # row[self.field_name] = self.replace_with_space(row[self.field_name])
        # row[self.field_name] = row[self.field_name].strip()
        # year, match = self.year(row[self.field_name])
        # # row['year'] = year
        # if match:
        #     row[self.field_name] = row[self.field_name].replace(match.group(), '')
        # row[self.field_name] = row[self.field_name].strip()
        # row[self.field_name] = re.sub(r'\s+', ' ', row[self.field_name])
        # row[self.field_name] = row[self.field_name] if row[self.field_name] else None
        # row['price'] = self.price(row['price'])
        return row
 
    def normalize_data(self):
        self.data = self.data.progress_apply(self.normalize, axis=1)
        return self.data
