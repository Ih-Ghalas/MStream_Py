from anom import * 
import numpy as np
import pandas as pd
from river import stream, metrics, preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

data = pd.read_csv(r'data\unswnb15.csv')

columns = data.columns
categ_label_time = ['1', '3', '5', '6', '14', '36', '39', '48','29', 'timestamp']
numer_features = [item for item in columns if item not in categ_label_time]

new = data[numer_features]
std = StandardScaler()
new = new[: 256]     

new = std.fit_transform(new.values)
pca = PCA(n_components=12)
pca.fit(new)

categ_features = ['1', '3', '5', '6', '14', '36', '39']

def preprocess(x):
    categ = []
    numeric = []
    for col in categ_features:
        categ.append(int(x.pop(col)))
    label = x.pop('48')
    timestamp = abs(float(x.pop('timestamp')))
    x.pop('29')
    for _, val in x.items():     
        numeric.append(float(val))
    numeric = np.array(numeric)
    numeric = std.transform(numeric.reshape(1, -1))
    numeric = pca.transform(numeric)
    return numeric[0], categ, timestamp, label

num_rows = 2
num_buckets = 1024
factor = 0.4
dimension1 = 12
dimension2 = 7

anomaly_detector = MStream(num_rows, num_buckets, factor, dimension1, dimension2)

auc = metrics.ROCAUC()
scaler = preprocessing.Normalizer(order=2)
for x, _ in stream.iter_csv(r'data\unswnb15.csv'):
    numeric, categ, timestamp, y = preprocess(x)
    numeric = {k:val for k,val in enumerate(numeric)}
    numeric = scaler.transform_one(numeric)
    numeric = list(numeric.values())
    score = anomaly_detector.score_one(numeric, categ, timestamp)
    auc = auc.update((y!='normal'), score)
    print(score, y, auc)

print('AUC {}'.format(auc))
