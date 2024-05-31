#!/usr/bin/env python
# coding: utf-8

# # Data Preparation

# In[96]:


# Importing Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


# In[97]:


# Loading the Dataset
titanic = pd.read_csv(r'C:\Users\lenovo\Downloads\Titanic-Dataset.csv')
titanic


# In[99]:


# Reading first 5 rows
titanic.head()


# In[100]:


# Reading last 5 rows
titanic.tail()


# In[101]:


# Showing no. of rows and columns of dataset
titanic.shape


# In[102]:


# checking for columns
titanic.columns


# # Data Preprocessing and Data Cleaning

# In[103]:


# Checking for data types
titanic.dtypes


# In[104]:


# checking for duplicated values
titanic.duplicated().sum()


# In[105]:


# checking for null values
nv = titanic.isna().sum().sort_values(ascending=False)
nv = nv[nv>0]
nv


# In[106]:


# Cheecking what percentage column contain missing values
titanic.isnull().sum().sort_values(ascending=False)*100/len(titanic)


# In[107]:


# Since Cabin Column has more than 75 % null values .So , we will drop this column
titanic.drop(columns = 'Cabin', axis = 1, inplace = True)
titanic.columns


# In[108]:


# Filling Null Values in Age column with mean values of age column
titanic['Age'].fillna(titanic['Age'].mean(),inplace=True)

# filling null values in Embarked Column with mode values of embarked column
titanic['Embarked'].fillna(titanic['Embarked'].mode()[0],inplace=True)
# checking for null values
titanic.isna().sum()


# In[109]:


# Finding no. of unique values in each column of dataset
titanic[['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Embarked']].nunique().sort_values()


# In[110]:


titanic['Survived'].unique()


# In[111]:


titanic['Sex'].unique()


# In[112]:



titanic['Pclass'].unique()


# In[113]:


titanic['SibSp'].unique()


# In[114]:


titanic['Parch'].unique()


# In[115]:


titanic['Embarked'].unique()


# # Dropping Some Unnecessary Columns

# There are 3 columns i.e.. PassengerId, Name , Ticket are unnecessary columns which have no use in data modelling . So, we will drop these 3 columns

# In[116]:


titanic.drop(columns=['PassengerId','Name','Ticket'],axis=1,inplace=True)
titanic.columns


# In[117]:


# Showing inforamation about the dataset
titanic.info()


# In[118]:


# showing info. about numerical columns
titanic.describe()


# In[119]:


# showing info. about categorical columns
titanic.describe(include='O')


# # Data Visualization

# 1. Sex Column

# In[120]:


d1 = titanic['Sex'].value_counts()
d1


# In[121]:


# Plotting Count plot for sex column
sns.countplot(x=titanic['Sex'])
plt.show()


# In[122]:


# Plotting Percantage Distribution of Sex Column
plt.figure(figsize=(5,5))
plt.pie(d1.values,labels=d1.index,autopct='%.2f%%')
plt.legend()
plt.show()


# In[123]:


# Showing Distribution of Sex Column Survived Wise
sns.countplot(x=titanic['Sex'],hue=titanic['Survived']) # In Sex (0 represents female and 1 represents male)
plt.show()


# This plot clearly shows male died more than females and females survived more than male percentage.

# In[124]:


# Showing Distribution of Embarked Sex wise
sns.countplot(x=titanic['Embarked'],hue=titanic['Sex'])
plt.show()


# 2. Pclass Column

# In[125]:


# Plotting CountPlot for Pclass Column
sns.countplot(x=titanic['Pclass'])
plt.show()


# In[126]:


# Showing Distribution of Pclass Sex wise
sns.countplot(x=titanic['Pclass'],hue=titanic['Sex'])
plt.show()


# In[127]:


# Age Distribution
sns.kdeplot(x=titanic['Age'])
plt.show()


# From this plot it came to know that most of the people lie between 20-40 age group.
# 
# Analysing Target Variable
# 
# Survived Column

# In[128]:


# Plotting CountPlot for Survived Column
print(titanic['Survived'].value_counts())
sns.countplot(x=titanic['Survived'])
plt.show()


# This plot Clearly shows most people are died

# In[129]:


# Showing Distribution of Parch Survived Wise
sns.countplot(x=titanic['Parch'],hue=titanic['Survived'])
plt.show()


# In[130]:


# Showing Distribution of SibSp Survived Wise
sns.countplot(x=titanic['SibSp'],hue=titanic['Survived'])
plt.show()


# In[131]:


# Showing Distribution of Embarked Survived wise
sns.countplot(x=titanic['Embarked'],hue=titanic['Survived'])
plt.show()


# In[132]:


# Showinf Distribution of Age Survived Wise
sns.kdeplot(x=titanic['Age'],hue=titanic['Survived'])
plt.show()


# This Plot showing most people of age group of 20-40 are died

# In[133]:


# Plotting Histplot for Dataset
titanic.hist(figsize=(10,10))
plt.show()


# In[134]:


# Plotting Boxplot for dataset
# Checking for outliers
sns.boxplot(titanic)
plt.show()


# This Plot showing Outliers in 2 columns i.e.. Age and Fare.

# In[135]:


# showing Correlation
titanic.corr()


# In[136]:


# Showing Correlation Plot
sns.heatmap(titanic.corr(),annot=True,cmap='coolwarm')
plt.show()


# This Plot is clearly showing
# 
# 1. Strong Positive Correlation between SibSp and Parch
# 2. Strong Negative Correlation between Pclass and Fare

# In[138]:


# Plotting pairplot
sns.pairplot(titanic)
plt.show()


# # Checking the target variable

# In[139]:


titanic['Survived'].value_counts()


# In[140]:


sns.countplot(x=titanic['Survived'])
plt.show()


# # Label Encoding

# In[141]:


from sklearn.preprocessing import LabelEncoder
# Create an instance of LabelEncoder
le = LabelEncoder()

# Apply label encoding to each categorical column
for column in ['Sex','Embarked']:
    titanic[column] = le.fit_transform(titanic[column])

titanic.head()

# Sex Column

# 0 represents female
# 1 represents Male

# Embarked Column

# 0 represents C
# 1 represents Q
# 2 represents S


# # Data Modelling

# In[142]:


# importing libraries

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score


# # Selecting the independent and dependent Features

# In[143]:


cols = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
x = titanic[cols]
y = titanic['Survived']
print(x.shape)
print(y.shape)
print(type(x))  # DataFrame
print(type(y))  # Series


# In[144]:


x.head()


# In[145]:


y.head()


# # Train_Test_Split

# In[146]:


print(891*0.10)


# In[147]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.10,random_state=1)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# # Creating Functions to compute Confusion Matrix, Classification Report and to generate Training and the Testing Score(Accuracy)

# In[148]:


def cls_eval(ytest,ypred):
    cm = confusion_matrix(ytest,ypred)
    print('Confusion Matrix\n',cm)
    print('Classification Report\n',classification_report(ytest,ypred))

def mscore(model):
    print('Training Score',model.score(x_train,y_train))  # Training Accuracy
    print('Testing Score',model.score(x_test,y_test))     # Testing Accuracy


# 1. Logistic Regression

# In[149]:


# Building the logistic Regression Model
lr = LogisticRegression(max_iter=1000,solver='liblinear')
lr.fit(x_train,y_train)


# In[150]:


# Computing Training and Testing score
mscore(lr)


# In[151]:


# Generating Prediction
ypred_lr = lr.predict(x_test)
print(ypred_lr)


# In[152]:


# Evaluate the model - confusion matrix, classification Report, Accuracy score
cls_eval(y_test,ypred_lr)
acc_lr = accuracy_score(y_test,ypred_lr)
print('Accuracy Score',acc_lr)


# 2. knn Classifier Model

# In[153]:


# Building the knnClassifier Model
knn=KNeighborsClassifier(n_neighbors=8)
knn.fit(x_train,y_train)


# In[154]:


# Computing Training and Testing score
mscore(knn)


# In[155]:


# Generating Prediction
ypred_knn = knn.predict(x_test)
print(ypred_knn)


# In[156]:


# Evaluate the model - confusion matrix, classification Report, Accuracy score
cls_eval(y_test,ypred_knn)
acc_knn = accuracy_score(y_test,ypred_knn)
print('Accuracy Score',acc_knn)


# 3. SVC

# In[157]:


# Building Support Vector Classifier Model
svc = SVC(C=1.0)
svc.fit(x_train, y_train)


# In[158]:


# Computing Training and Testing score
mscore(svc)


# In[159]:


# Generating Prediction
ypred_svc = svc.predict(x_test)
print(ypred_svc)


# In[160]:


# Evaluate the model - confusion matrix, classification Report, Accuracy score
cls_eval(y_test,ypred_svc)
acc_svc = accuracy_score(y_test,ypred_svc)
print('Accuracy Score',acc_svc)


# 4. Random Forest Classifier

# In[161]:


# Building the RandomForest Classifier Model
rfc=RandomForestClassifier(n_estimators=80,criterion='entropy',min_samples_split=5,max_depth=10)
rfc.fit(x_train,y_train)


# In[162]:


# Computing Training and Testing score
mscore(rfc)


# In[163]:


# Generating Prediction
ypred_rfc = rfc.predict(x_test)
print(ypred_rfc)


# In[164]:



# Evaluate the model - confusion matrix, classification Report, Accuracy score
cls_eval(y_test,ypred_rfc)
acc_rfc = accuracy_score(y_test,ypred_rfc)
print('Accuracy Score',acc_rfc)


# 5. DecisionTree Classifier

# In[165]:


# Building the DecisionTree Classifier Model
dt = DecisionTreeClassifier(max_depth=5,criterion='entropy',min_samples_split=10)
dt.fit(x_train, y_train)


# In[166]:


# Computing Training and Testing score
mscore(dt)


# In[167]:


# Generating Prediction
ypred_dt = dt.predict(x_test)
print(ypred_dt)


# In[168]:


# Evaluate the model - confusion matrix, classification Report, Accuracy score
cls_eval(y_test,ypred_dt)
acc_dt = accuracy_score(y_test,ypred_dt)
print('Accuracy Score',acc_dt)


# 6. Adaboost Classifier

# In[169]:


# Builing the Adaboost model
ada_boost  = AdaBoostClassifier(n_estimators=80)
ada_boost.fit(x_train,y_train)


# In[170]:


# Computing the Training and Testing Score
mscore(ada_boost)


# In[171]:


# Generating the predictions
ypred_ada_boost = ada_boost.predict(x_test)


# In[172]:


# Evaluate the model - confusion matrix, classification Report, Accuracy Score
cls_eval(y_test,ypred_ada_boost)
acc_adab = accuracy_score(y_test,ypred_ada_boost)
print('Accuracy Score',acc_adab)


# In[173]:



models = pd.DataFrame({
    'Model': ['Logistic Regression','knn','SVC','Random Forest Classifier','Decision Tree Classifier','Ada Boost Classifier'],
    'Score': [acc_lr,acc_knn,acc_svc,acc_rfc,acc_dt,acc_adab]})

models.sort_values(by = 'Score', ascending = False)


# In[174]:


colors = ["blue", "green", "red", "yellow","orange","purple"]

sns.set_style("whitegrid")
plt.figure(figsize=(15,5))
plt.ylabel("Accuracy %")
plt.xlabel("Algorithms")
sns.barplot(x=models['Model'],y=models['Score'], palette=colors )
plt.show()

