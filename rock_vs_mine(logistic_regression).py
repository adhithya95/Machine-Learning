import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
df = pd.read_csv('/content/sonar data.csv',header=None)
x = df.drop(columns=60,axis=1)
y = df[60]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
model = LogisticRegression()
model.fit(x_train,y_train)
x_pred = model.predict(x_train)
acc1 = accuracy_score(x_pred,y_train)
#training data accuracy
print(acc1)
x_test_pred = model.predict(x_test)
acc2 = accuracy_score(x_test_pred,y_test)
#testing_data accuracy
print(acc2)
input_data = (0.0200,0.0371,0.0428,0.0207,0.0954,0.0986,0.1539,0.1601,0.3109,0.2111,0.1609,0.1582,0.2238,0.0645,0.0660,0.2273,0.3100,0.2999,0.5078,0.4797,0.5783,0.5071,0.4328,0.5550,0.6711,0.6415,0.7104,0.8080,0.6791,0.3857,0.1307,0.2604,0.5121,0.7547,0.8537,0.8507,0.6692,0.6097,0.4943,0.2744,0.0510,0.2834,0.2825,0.4256,0.2641,0.1386,0.1051,0.1343,0.0383,0.0324,0.0232,0.0027,0.0065,0.0159,0.0072,0.0167,0.0180,0.0084,0.0090,0.0032)
input_data_to_nparray = np.asarray(input_data)
reshaped_data = input_data_to_nparray.reshape(1,-1)
result = model.predict(reshaped_data)
print(result)
