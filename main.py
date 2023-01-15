from anom import * 
import numpy as np
import pandas as pd
from river import stream, metrics, preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('dataset.csv')
# getting the numerical features
columns = data.columns
categ_label =  ['normal.','udp_cat','private_cat','SF_cat']   
numer_features = [item for item in columns if item not in categ_label]
categ_features = ['udp_cat','private_cat','SF_cat']

# Initializing the Standard Scaler and PCA
std = StandardScaler()
pca = PCA(n_components=12)       

# keeping only the numerical features to apply PCA 
new = data[numer_features]
new = new[: 256]     
new = std.fit_transform(new.values)
pca.fit(new)


def preprocess(x, i):
    categ = []
    numeric = []
    for col in categ_features:
        categ.append(int(x.pop(col)))
    label = x.pop('normal.')
    for _, val in x.items():     
        numeric.append(float(val))
    numeric = np.array(numeric)
    numeric = std.transform(numeric.reshape(1, -1))
    numeric = pca.transform(numeric)[0]
    return numeric, categ, i, label


num_rows = 2
num_buckets = 1024
factor = 0.85
dimension1 = 12
dimension2 = 3

anomaly_detector = MStream(num_rows, num_buckets, factor, dimension1, dimension2)

i=1
auc = metrics.ROCAUC()
scaler = preprocessing.Normalizer(order=2)

for x, _ in stream.iter_csv('dataset.csv'):
    numeric, categ, j, y = preprocess(x, 1 + i/1000)

    numeric = {k:val for k,val in enumerate(numeric)}
    numeric = scaler.transform_one(numeric)
    numeric = list(numeric.values())
    score = anomaly_detector.score_one(numeric, categ, j)
    auc = auc.update((y!='normal.'), score)
    print(score, y, auc)
    i+=1

print('AUC {}'.format(auc))