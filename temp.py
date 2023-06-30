import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
dataset= pd.read_csv('Credit_Card_Applications.csv')
X= dataset.iloc[:, :-1].values
Y= dataset.iloc[:, -1].values
from sklearn.preprocessing import MinMaxScaler
sc= MinMaxScaler(feature_range= (0,1))
X= sc.fit_transform(X)