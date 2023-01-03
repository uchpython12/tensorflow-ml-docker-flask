# -*- coding: UTF-8 -*-
import pickle
"""
import gzip
# 載入gzip Model
# with gzip.open('./model/xgboost-iris.pgz', 'rb') as f:
"""

# 載入 Model 如果要載入pgz將上面備註替換即可
with open('./model/xgboost-iris.pickle', 'rb') as f:
    xgboostModel = pickle.load(f)

def predict(input):
    pred=xgboostModel.predict(input)[0]
    print(pred)
    return pred


