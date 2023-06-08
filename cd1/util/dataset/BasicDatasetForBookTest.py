import pandas as pd
import numpy as np
import torch

class BasicDatasetForBookTest:
    def __init__(self, data) -> None:
        self._data = data
    
    @classmethod
    def load(cls, file_path):
        data = pd.read_csv(file_path)
        draft_data = []
        
        # Title, Image
        
        for title,caption in zip(data['Title'],data['caption']): 
            draft_data.append((title,str(caption)))
            
        return cls(draft_data)

    def __getitem__(self, index) -> dict:
        title,caption = self._data[index]
        
        return {
            'title':title,
            'caption':caption
        }
        
    def __len__(self):
        return len(self._data)