import redis
import sqlite3 as sql
import numpy as np
import time

import psycopg2
import json
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt

########## 1: read raw data from database (Postgresql) and save it on Redis ##########

### read table from PostgreSQL ###
conn = psycopg2.connect(host="127.0.0.1",database="chilindo", user="postgres", password="password")
c = conn.cursor()
c.execute('SELECT * FROM pokemon;')
columns = ('no', 'name', 'type1', 'type2', 'total', 'hp', 'attack', 'defense', 'special_attack', 'special_defense', 'speed', 'generation', 'legendary')
results = []
r = redis.Redis(host='127.0.0.1', port="6379", db=0)

for row in c.fetchall():
    results.append(dict(zip(columns, row)))

### save it on Redis ###
p = pickle.dumps(results)
r.set('pokemon', p)

######################################################################################

########## 2: get raw data from Redis, analyze and present final value on screen ##########

### get raw data from Redis ###
data = pickle.loads(r.get('pokemon'))

for d in data:
    nx = np.array([d['total'], d['hp']])
    if(d['legendary'] is True):
        l = 1
    else:
        l = 0
    ny = np.array(l)
    if 'X' in locals():
        X = np.vstack([X, nx])
        y = np.append(y, l)
    else:
        X = nx
        y = np.array([l])

### implement cross validation (Training/Validation - 75/25) ###
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

### LogisticRegression ###
LogReg = LogisticRegression()
LogReg.fit(X_train, y_train)
y_score = LogReg.predict(X_test)
print('Score:', LogReg.score(X_test, y_test))

### Plot an ROC curve with your model outcome ###
fpr, tpr, thresholds = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)
print('FPR:', fpr)
print('TPR:', tpr)
print('THRESHOLDS:', thresholds)
print('ROC_AUC:', roc_auc)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

### Calculate the True Positive Rate, False Negative Rate and Precision & Store these values in redis using the name of the values(TruePositiveRate, FalseNegativeRate, Precision) as the key values. ###
tn, fp, fn, tp = confusion_matrix([0, 1, 0, 1], [1, 1, 1, 0]).ravel()
r.set('TruePositiveRate', tp)
r.set('FalseNegativeRate', fn)
r.set('Precision', tp/(tp+fp))

###########################################################################################