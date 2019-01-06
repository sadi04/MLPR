import pandas as pd
import numpy as np
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import LabelEncoder
import math

dataframe = pd.read_csv("online1_data.csv")


dataframe.drop(columns=['age','fnlwgt','education-num','capital-gain','capital-loss', 'hours-per-week'],inplace=True)
dataframe.dropna(inplace=True)
