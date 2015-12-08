'''
This benchmark uses xgboost and early stopping to achieve a score of 0.38019
In the liberty mutual group: property inspection challenge

Based on Abhishek Catapillar benchmark
https://www.kaggle.com/abhishek/caterpillar-tube-pricing/beating-the-benchmark-v1-0

@author Devin

Have fun;)
'''

# Estudando codigo de brasileiro em 2o lugar no ranking do kaggle para esta competicao para aprender python
# https://www.kaggle.com/titericz/liberty-mutual-group-property-inspection-prediction/done-done-3
## Autor: Gilberto Titericz Junior

import pandas as pd
import numpy as np 
from sklearn import preprocessing
import xgboost as xgb
