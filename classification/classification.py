import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import numpy as np


path = os.getcwd()

df = pd.read_excel(path + "/highlow.xlsx")

# print(df.values.shape)

label = np.array(df['高/低危组'])
# data = df[['前SIRI', "前LMR", '前PLR', '前NLR']]
data = np.array(df['前SIRI']).reshape(-1, 1)

x_train_, x_test_, y_train, y_test = train_test_split(data, label, test_size=0.5)

sc = StandardScaler()
sc.fit(x_train_)
x_train = sc.transform(x_train_)
x_test = sc.transform(x_test_)

print(sc.__dict__)

logreg = LogisticRegression()
logreg.fit(x_train, y_train)

print("class: ", logreg.classes_)
print("weighes: ", logreg.coef_, logreg.intercept_)

y_pred = logreg.predict(x_test)
print("accuracy: ", accuracy_score(y_test, y_pred))

logit_roc_auc = roc_auc_score(y_test, y_pred)
fpr, tpr, thre = roc_curve(y_test, logreg.predict_proba(x_test)[:,1])
maxindex = (tpr - fpr).tolist().index(max(tpr - fpr))
threshold = thre[maxindex]
print("threshold: ", threshold)

plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.3f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.scatter(fpr[maxindex], tpr[maxindex])
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()