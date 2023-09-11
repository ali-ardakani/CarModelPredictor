import pandas as pd
import random
import tqdm

class NoiseGenerator:
    
    @staticmethod
    def get_noisy_words(src1: pd.Series, src2: pd.Series) -> list:
        """
        This function takes two pandas Series objects as input and returns a list of unique words that appear in the first
        Series but not in the second. The input Series objects should contain strings that can be split into words using
        whitespace as a delimiter.
        
        Args:
            src1 (pd.Series): The first pandas Series object.
            src2 (pd.Series): The second pandas Series object.
        
        Returns:
            list: A list of unique words that appear in `src1` but not in `src2`.
        """
        return list(set.union(*(src1.str.split().apply(set) - src2.str.split().apply(set))))
    
    @staticmethod
    def generate_noise_txt(txt: str, noise: list, range_word: tuple = (1, 5)) -> str:
        """
        This function takes a string and a list of words as input and returns a string with some of the words in the
        input string replaced with random words from the list of words.
         
        Args:
            txt (str): The input string.
            noise (list): A list of words.
            range_word (tuple, optional): A tuple of two integers that represent the range of the number of words to be
                replaced. Defaults to (1, 5).
                
        Returns:
            str: A string with some of the words in the input string replaced with random words from the list of words.
        """
        return txt + ' '.join(random.sample(noise, random.randint(*range_word)))

    @staticmethod
    def generate_noisy_df(
        df: pd.DataFrame,
        add_noise_to: str,
        group_by: str, 
        noise: list, 
        limit: int, 
        range_samples: tuple,
        range_word: tuple = (1, 5)) -> pd.DataFrame:
        """
        Generate noisy data for a given DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to add noise to.
            add_noise_to (str): The column name to add noise to.
            group_by (str): The column name to group by.
            noise (list): The list of noise types to add.
            limit (int): The minimum number of samples in a group to add noise to.
            range_samples (tuple): The range of samples to add noise to.
            range_word (tuple, optional): The range of words to add noise to. Defaults to (1, 5).

        Returns:
            pd.DataFrame: The DataFrame with added noise.
        """
        df = df.copy()
        groups_needed = df.groupby(group_by).filter(lambda x: len(x) < limit)[group_by].unique()
        for group in tqdm.tqdm(groups_needed):
            base = df[df[group_by] == group]
            new_samples = base.sample(random.randint(*range_samples), replace=True)
            new_samples[add_noise_to] = new_samples[add_noise_to].apply(
                lambda x: NoiseGenerator.generate_noise_txt(x, noise, range_word))
            df = pd.concat([df, new_samples], axis=0, ignore_index=True)
        return df