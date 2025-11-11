# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Preprocess Data: Clean emails and convert text to numeric features (TF-IDF or BoW).
2. Split Dataset: Divide data into training and testing sets.
3. Train Model: Use SVM classifier (linear/RBF kernel) on training data.
4. Evaluate Model: Test performance using accuracy, precision, recall, and F1-score.

## Program:
```
Program to implement the SVM For Spam Mail Detection..
Developed by: MALENI M
RegisterNumber: 212223040110
```

```PYTHON
import chardet
file="spam.csv"
with open(file,'rb') as rawdata:
    result=chardet.detect(rawdata.read(100000))
result
```
<img width="1299" height="92" alt="image" src="https://github.com/user-attachments/assets/27e7f6c1-76f1-4bf4-ae96-b7e58ca44028" />

```PYTHON
import pandas as pd
data=pd.read_csv('spam.csv',encoding='Windows-1252')
data.head()
```
<img width="1416" height="426" alt="image" src="https://github.com/user-attachments/assets/bbf99853-62bd-4e56-ba2a-9dbbc7ff995e" />


```PYTHON
data.isnull().sum()
```
<img width="497" height="260" alt="image" src="https://github.com/user-attachments/assets/b83be0ad-23da-4935-a820-63b18a06f425" />


```PYTHON
data.info()
```
<img width="765" height="456" alt="image" src="https://github.com/user-attachments/assets/2576a6e2-d1c6-4fdc-bd45-b270af6186c8" />


```PYTHON
x=data["v2"].values
y=data["v1"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
x_train
```
<img width="878" height="107" alt="image" src="https://github.com/user-attachments/assets/e65d8766-4a0a-4f69-8c00-d03eff9058b6" />

```PYTHON
x_test
```
<img width="884" height="151" alt="image" src="https://github.com/user-attachments/assets/a7d1e9ac-98c2-4c14-803d-67635cb6bf23" />

```PYTHON
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
x_train
```
<img width="1494" height="343" alt="image" src="https://github.com/user-attachments/assets/07c70bfd-73dc-48c7-b581-596c4f2a4ee9" />


```PYTHON
x_test
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
print("NAME: MALENI M")
print("REG NO:212223040110")
cr=metrics.classification_report(y_test,y_pred)
print("Classification report:")
print(cr)
cm=metrics.confusion_matrix(y_test,y_pred)
print("Confusion Matrix")
print(cm)
```

## Output:
<img width="3199" height="1799" alt="image" src="https://github.com/user-attachments/assets/24a866bb-ed99-4696-b087-9fa56227ddfc" />



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
