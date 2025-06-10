#this dataset contains two coloumns one is named as sentiment and the other as news sentiment 
#sentiment column contains two values either positive or negative 
#news column contains the news 
import kagglehub
import pandas as pd
import kagglehub.datasets
filepath = kagglehub.dataset_download("ankurzing/sentiment-analysis-for-financial-news")
def load_data(filepath):
    df=pd.read_csv(filepath,sep=';',names=['sentiment','news'])
    return df
def preprocess_text(text):
    return text.lower().strip()
#here in this file downloading the dataset from the kaggle website and then preprocessing the text 
#and then returning the dataframe











