#!/usr/bin/env python
# coding: utf-8

# ## Loan prediction

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


train_data =pd.read_csv('train_ctrUa4K.csv')
test_data =pd.read_csv('test_lAUu6dG.csv')
test_original =test_data.copy()


# In[3]:


train_data.head()


# In[4]:


test_data.head()


# In[5]:


train_data.isnull().sum()


# In[6]:


train_data.info()


# In[7]:


test_data.isnull().sum()


# In[ ]:





# ### Handling Missing Values in data

# In[8]:


value = "Unknown"
train_data['Gender'].fillna(value,inplace=True)
train_data['Married'].fillna(value,inplace=True)
train_data['Dependents'].fillna(0,inplace=True)
train_data['Self_Employed'].fillna(value,inplace=True)
train_data['Credit_History'].fillna(0.0,inplace=True)


# In[9]:


train_data.isnull().sum()


# In[10]:


value = "Unknown"
test_data['Gender'].fillna(value,inplace=True)
test_data['Dependents'].fillna(0,inplace=True)
test_data['Self_Employed'].fillna(value,inplace=True)
test_data['Credit_History'].fillna(0.0,inplace=True)


# In[11]:


test_data.isnull().sum()


# In[12]:


#plt.scatter(train_data['LoanAmount'],train_data['Credit_History'])
#plt.show()
print(train_data.groupby('Credit_History')['LoanAmount'].value_counts())


# In[13]:


print(test_data.groupby('Credit_History')['LoanAmount'].value_counts())


# In[14]:


sns.boxplot(x='Credit_History',y='LoanAmount',data=train_data)
plt.figure(figsize=(12,7))
plt.show()


# In[15]:


sns.boxplot(x='Credit_History',y='LoanAmount',data=test_data)
plt.figure(figsize=(12,7))
plt.show()


# In[16]:


test_data.groupby('Credit_History')['LoanAmount'].median()


# In[17]:


print(train_data.groupby('Credit_History')['Loan_Status'].count())


# In[18]:


sns.countplot(x='Credit_History',hue='Loan_Status',data=train_data)
plt.figure(figsize=(12,7))
plt.show()


# In[19]:



train_data['Credit_History'].unique()


# In[20]:


sns.countplot(x='Loan_Status',data=train_data)
plt.show()


# In[21]:


sns.countplot(x='Credit_History',data=train_data)
plt.show()


# In[22]:


sns.countplot(x='Credit_History',data=test_data)
plt.show()


# In[23]:


miss_val =train_data.loc[train_data['LoanAmount'].isna()]

miss_val.index


# In[24]:


def imputetrain_LoanAmount(cols):
    LoanAmount =cols[0]
    Credit_History=cols[1]
    if pd.isnull(LoanAmount):
       
         if Credit_History==1.0:
            return 128.0
         else:
            #return 127.0
            return 160.0
        
    return LoanAmount


# In[25]:


def imputetest_LoanAmount(cols):
    LoanAmount =cols[0]
    Credit_History=cols[1]
    if pd.isnull(LoanAmount):
       
         if Credit_History==1.0:
            return 125.0
         else:
            return 130.0
        
    return LoanAmount


# In[26]:


train_data['LoanAmount']=train_data[['LoanAmount','Credit_History']].apply(imputetrain_LoanAmount,axis=1)


# In[27]:


test_data['LoanAmount']=test_data[['LoanAmount','Credit_History']].apply(imputetrain_LoanAmount,axis=1)


# In[28]:


train_data.head()


# In[29]:


#train_data['Loan_Amount_Term'].unique()
train_data['Loan_Amount_Term'].value_counts()


# In[30]:


test_data['Loan_Amount_Term'].value_counts()


# In[31]:


sns.countplot(x='Loan_Amount_Term',data=train_data)
plt.figure(figsize=(12,7))
plt.show()


# In[32]:


train_data['Loan_Amount_Term'].fillna(360.0,inplace=True)
test_data['Loan_Amount_Term'].fillna(360.0,inplace=True)


# In[33]:


train_data.isnull().sum()


# In[34]:


test_data.isnull().sum()


# ### Handling Categorical features

# In[35]:


categ=[feature for feature in train_data.columns if train_data[feature].dtype=='O' ]


# In[36]:


categ


# In[37]:


num=[feature for feature in train_data.columns if train_data[feature].dtype!='O']


# In[38]:


num


# In[39]:


categ_test=[feature for feature in test_data.columns if test_data[feature].dtype=='O' ]


# In[40]:


num_test=[feature for feature in test_data.columns if test_data[feature].dtype!='O']


# In[41]:


for feature in categ:
    print(feature,'has : ','{} categories'.format(len(train_data[feature].unique())))


# In[42]:


train_data['Married'].unique()


# In[43]:


conv_to_num = {
                "Gender"  : {'Male': 0, 'Female': 1, 'Unknown' :2},
                "Married" : {'No' : 0, 'Yes' : 1, 'Unknown' :2},
              "Dependents" : {'0' : 0, '1' : 1, '2' : 2, '3+' : 3, 0 :0},
                "Education": {'Graduate' : 0, 'Not Graduate' : 1},
            "Self_Employed" : {'No' : 0, 'Yes' : 1, 'Unknown' : 2},
            "Property_Area" : {'Urban' : 0, 'Rural' :1, 'Semiurban' :2},
              "Loan_Status" : {'Y' : 1, 'N' : 0}
      
               }


# In[44]:


train_data = train_data.replace(conv_to_num)


# In[45]:


for feature in categ_test:
    print(feature,'has : ','{} categories'.format(len(test_data[feature].unique())))


# In[46]:


test_data['Property_Area'].unique()


# In[47]:


conv_test_num = {
                "Gender"  : {'Male': 0, 'Female': 1, 'Unknown' :2},
                "Married" : {'No' : 0, 'Yes' : 1},
              "Dependents" : {'0' : 0, '1' : 1, '2' : 2, '3+' : 3, 0 :0},
                "Education": {'Graduate' : 0, 'Not Graduate' : 1},
            "Self_Employed" : {'No' : 0, 'Yes' : 1, 'Unknown' : 2},
            "Property_Area" : {'Urban' : 0, 'Rural' :1, 'Semiurban' :2}                  
               }


# In[48]:


test_data = test_data.replace(conv_test_num)


# In[ ]:





# In[49]:


labels = train_data['Loan_ID'].unique()
id_len = range(0,len(labels))
train_data['Loan_ID'].replace(labels,id_len,inplace=True)


# In[50]:


train_data.head()


# In[51]:


labels = test_data['Loan_ID'].unique()
id_len = range(0,len(labels))
test_data['Loan_ID'].replace(labels,id_len,inplace=True)


# In[52]:


test_data.head()


# In[53]:


train_data.info()


# In[54]:


nx=sns.pairplot(data=train_data,hue='Loan_Status')
plt.figure(figsize=(8,7))
plt.show()


# In[55]:


ax = sns.heatmap(train_data.corr(),annot=True,cmap="YlGnBu")
plt.figure(figsize=(50.0,60.0))
plt.show()


# In[56]:


ax=sns.boxplot(x='Loan_Status',y='ApplicantIncome',data=train_data)
plt.figure(figsize=(12,7))


# In[57]:


test_data['ApplicantIncome'].plot.box()


# In[58]:


ax=sns.boxplot(x='Loan_Status',y='CoapplicantIncome',data=train_data)
plt.figure(figsize=(12,7))


# In[59]:


test_data['CoapplicantIncome'].plot.box()


# In[60]:


ax=sns.boxplot(x='Loan_Status',y='LoanAmount',data=train_data)
plt.figure(figsize=(12,7))


# In[61]:


test_data['LoanAmount'].plot.box()


# In[62]:


ax=sns.boxplot(x='Loan_Status',y='Loan_Amount_Term',data=train_data)
plt.figure(figsize=(12,7))


# In[63]:


ax=sns.boxplot(x='Loan_Status',y='Credit_History',data=train_data)
plt.figure(figsize=(12,7))


# In[64]:


ax=sns.scatterplot(x='LoanAmount',y='Loan_Amount_Term',data=train_data)
plt.figure(figsize=(12,7))


# In[65]:


ax=sns.scatterplot(x='LoanAmount',y='Loan_Amount_Term',data=test_data)
plt.figure(figsize=(12,7))


# In[66]:


ax=sns.scatterplot(x='ApplicantIncome',y='CoapplicantIncome',data=train_data)
plt.figure(figsize=(12,7))


# In[67]:


ax=sns.scatterplot(x='ApplicantIncome',y='CoapplicantIncome',data=test_data)
plt.figure(figsize=(12,7))


# # Treating Outliers 

# In[68]:


for feature in ['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term']:
    if 0 in train_data[feature].unique():
        pass
    else:
        train_data[feature]=np.log(train_data[feature])


# In[69]:


for feat in ['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term']:
    if 0 in test_data[feat].unique():
        pass
    else:
        test_data[feat]=np.log(test_data[feat])


# In[70]:


train_data.describe().transpose()


# In[71]:


ax=sns.boxplot(x='Loan_Status',y='LoanAmount',data=train_data)
plt.figure(figsize=(12,7))


# In[72]:


ax=sns.boxplot(x='Loan_Status',y='ApplicantIncome',data=train_data)
plt.figure(figsize=(12,7))


# In[73]:


ax=sns.boxplot(x='Loan_Status',y='CoapplicantIncome',data=train_data)
plt.figure(figsize=(12,7)) 


# In[74]:


train_data.head()


# In[75]:


test_data.head()


# ## Training the Model

# ### KNN

# In[76]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


# In[77]:


y =train_data['Loan_Status']
x =train_data.drop(['Loan_ID','Loan_Status'],axis=1)


# In[78]:


id_list =test_data['Loan_ID']


# In[79]:


xtestfinal =test_data.drop(['Loan_ID'],axis=1)


# In[80]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=1)


# In[81]:


knn =KNeighborsClassifier()


# In[82]:


knn.fit(x_train,y_train)


# In[83]:


y_test_pred = knn.predict(x_test)


# In[84]:


from sklearn.metrics import accuracy_score
score = accuracy_score(y_test,y_test_pred)


# In[85]:


score


# In[86]:


from sklearn.model_selection import RandomizedSearchCV


# In[87]:


params = {'n_neighbors': [2,3,5] ,
             'weights' : ['uniform','distance'],
           'algorithm' : ['auto','kd_tree'],
            'leaf_size': [30,40,50]
         }


# In[88]:


rs =RandomizedSearchCV(estimator = knn,param_distributions = params, n_iter=10,n_jobs=-1,scoring='accuracy')


# In[89]:


rs.fit(x_train,y_train)


# In[90]:


rs.best_params_


# In[91]:


knn1 =KNeighborsClassifier(algorithm='kd_tree', leaf_size=50, n_neighbors=3)


# In[92]:


knn1.fit(x_train,y_train)


# In[93]:


y_pred = knn1.predict(x_test)


# In[94]:


from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

acc_score = metrics.accuracy_score(y_test,y_pred)
avg_precision_score =metrics.average_precision_score(y_test,y_pred)
f1_score=metrics.f1_score(y_test,y_pred)
precision=metrics.precision_score(y_test,y_pred)
recall=metrics.recall_score(y_test,y_pred)
roc=metrics.roc_auc_score(y_test,y_pred)

print('Accuracy score :',acc_score)
print('Avg.Precision score :',avg_precision_score)
print('f1 score :',f1_score)
print('Precision :',precision)
print('Recall :',recall)
print('ROC Score :',roc)


# In[95]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
kfold_val =KFold(10)


# In[96]:


result_kfold=cross_val_score(knn,x,y,cv=kfold_val)
print(result_kfold)


# In[97]:


print('Mean value - KNN ',': {}'.format(np.mean(result_kfold)))


# #### Model Prediction for Test Data 

# In[98]:


y_predfinal =knn.predict(xtestfinal)


# In[99]:


y_predfinal.shape


# ### Decision Tree 

# In[100]:


from sklearn import tree


# In[101]:


dt =tree.DecisionTreeClassifier(random_state=1)


# In[102]:


dt.fit(x_train,y_train)


# In[103]:


y_dtpred =dt.predict(x_test)


# In[104]:


print('Accuracy score for Decision Tree:',metrics.accuracy_score(y_test,y_dtpred))


# In[105]:


kfold_dt =KFold(10)
result_kfold_dt=cross_val_score(dt,x,y,cv=kfold_dt)
print(result_kfold_dt)


# In[106]:


print('Mean value - Decision Tree ',': {}'.format(np.mean(result_kfold_dt)))


# In[107]:


ypred_final=dt.predict(xtestfinal)


# In[108]:


ypred_final


# ### Using Random Forest Classifier

# In[109]:


from sklearn.ensemble import RandomForestClassifier


# In[110]:


rfc =RandomForestClassifier()


# In[111]:


rfc.fit(x_train,y_train)


# In[112]:


y_rfcpred= rfc.predict(x_test)


# In[113]:


print('Accuracy score for Random Forest:',metrics.accuracy_score(y_test,y_rfcpred))


# In[114]:


kfold_rf =KFold(10)
result_kfold_rf=cross_val_score(rfc,x,y,cv=kfold_rf)
print(result_kfold_rf)


# In[115]:


print('Mean value - Random Forest Classifier ',': {}'.format(np.mean(result_kfold_rf)))


# In[116]:


y_rfcfinal= rfc.predict(xtestfinal)
y_rfcfinal


# In[117]:


params ={ 'n_estimators': [50,100,150,200],
          'criterion': ['gini','entropy'],
          'max_depth': [2,4,6],
                   
        }


# In[118]:


rs=RandomizedSearchCV(estimator=rfc,param_distributions=params,cv=5)


# In[119]:


rs.fit(x_train,y_train)


# In[120]:


rs.best_estimator_


# In[121]:


rfc1 =RandomForestClassifier(max_depth=4, n_estimators=150)


# In[122]:


rfc1.fit(x_train,y_train)


# In[123]:


y_rfc1pred=rfc1.predict(x_test)


# In[124]:


print('Accuracy score for Random Forest:',metrics.accuracy_score(y_test,y_rfc1pred))


# In[125]:


kfold_rf =KFold(10)
result_kfold_rfc1=cross_val_score(rfc1,x,y,cv=kfold_rf)
print(result_kfold_rfc1)


# In[126]:


print('Mean value - Random Forest Classifier ',': {}'.format(np.mean(result_kfold_rfc1)))


# In[127]:


ypred_final=rfc1.predict(xtestfinal)


# In[128]:


ypred_final


# ### Xgboost

# In[129]:


import xgboost 
from xgboost import XGBClassifier


# In[130]:


xgb =XGBClassifier()


# In[131]:


xgb.fit(x_train,y_train)


# In[132]:


y_xgbpred = xgb.predict(x_test)


# In[133]:


print('Accuracy score for XGboost:',metrics.accuracy_score(y_test,y_xgbpred))


# In[134]:


kfold_xg =KFold(10)
result_kfold_xg=cross_val_score(xgb,x,y,cv=kfold_xg)
print(result_kfold_xg)


# In[135]:


print('Mean value - Random Forest Classifier ',': {}'.format(np.mean(result_kfold_xg)))


# In[136]:


ypred_final=xgb.predict(xtestfinal)


# In[137]:


ypred_final


# # Final Analysis shows that RandomForestClassifier yields better results.

# ### The 

# In[138]:


print('Mean Score values after KFold :')
print(' KNN : {}'.format(np.mean(result_kfold)))
print(' Decision Tree : {}'.format(np.mean(result_kfold_dt)))
print(' Random Forest Classifier-before hyperparameter tuning : {}'.format(np.mean(result_kfold_rf)))
print(' Random Forest Classifier-after hyperparameter tuning : {}'.format(np.mean(result_kfold_rfc1)))
print(' XGBoost : {}'.format(np.mean(result_kfold_xg)))


# In[139]:


rfc1.feature_importances_


# In[140]:


imp_features =pd.Series(rfc1.feature_importances_,index=x.columns)
imp_features.plot(kind ='barh',figsize=(12,8))


# In[141]:


final= pd.read_csv('sample_submission_49d68Cx.csv')


# In[142]:


final.head()


# In[143]:


ypredfinal=rfc1.predict(xtestfinal)
final['Loan_ID']=test_original['Loan_ID']
final['Loan_Status']=ypredfinal


# In[144]:


final['Loan_Status'].replace(0,'N',inplace=True)
final['Loan_Status'].replace(1,'Y',inplace=True)


# In[145]:


pd.DataFrame(final,columns=['Loan_ID','Loan_Status']).to_csv('RandomForest_output.csv')


# In[ ]:




