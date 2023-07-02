from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import pickle
import time


df = pd.read_csv('assets/feature.csv')
x = df.drop(['id'], axis=1)
y = df['id']
print(x)
print(y)

model = KNeighborsClassifier(n_neighbors=1)
model_time = time.time()
model.fit(x.values,y)
model_time = time.time() - model_time
print(f'Waktu Training Model : {model_time}')
# filename = 'assets/model.sav'
# pickle.dump(model, open(filename, 'wb'))


# feature_pred = [112.0044641967453, 72.56031973468694, 70.2353187506115, 106.21205204683694, 114.49454135459908, 52.009614495783374]
# feature_pred = np.array(feature_pred)
# feature_pred = feature_pred.reshape(1, -1)
# feature_pred = model.predict(feature_pred)
# print(feature_pred.shape)
# print(type(feature_pred))
# print(feature_pred[0])

