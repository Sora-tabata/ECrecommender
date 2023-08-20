import time
import pandas as pd
import numpy as np
from surprise import Dataset, Reader, accuracy
from surprise.model_selection import train_test_split
from surprise import (
    NormalPredictor, BaselineOnly, KNNBasic, KNNWithMeans, KNNWithZScore, 
    KNNBaseline, SVD, SVDpp, NMF, SlopeOne, CoClustering
)

data_ml_100k = Dataset.load_builtin(name=u'ml-100k', prompt=False)
data_ml_1m = Dataset.load_builtin(name=u'ml-1m', prompt=False)

ML100K_URL = 'http://files.grouplens.org/datasets/movielens/ml-100k/u.data'
df_custom = pd.read_csv(
    ML100K_URL, names=["userid", "itemid", "rating", "timestamp"], sep="\t"
)
reader = Reader(rating_scale=(1, 5))
data_custom = Dataset.load_from_df(
    df_custom[["userid", "itemid", "rating"]], reader
)

trainset, testset = train_test_split(data_custom, test_size=.2)

## test SVD model

algo = SVD()
algo.fit(trainset)
pred = algo.test(testset)
#pred = algo.fit(trainset_all).test(testset_all)
accuracy.rmse(pred),accuracy.mse(pred),accuracy.mae(pred),accuracy.fcp(pred)