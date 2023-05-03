# Hyperparameter Optimization for Logistic Regression Algorithms

# In[1]:


# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, auc,roc_curve
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay
from sklearn.metrics import ConfusionMatrixDisplay, classification_report


# In[2]:


# Preview the dataset
df = pd.read_csv('E:\diabetes.csv')
df.head()


# In[3]:


# View dimensions of dataset   
rows, col = df.shape
print ("Dimensions of dataset: {}" . format (df.shape))
print ('Rows:', rows, '\nColumns:', col)


# In[4]:


# Columns
df.columns


# In[5]:


# Information about the dataframe
df.info()


# In[6]:


# Statistical details
df.describe()


# In[7]:


# Checking the missing values
missing_values = pd.DataFrame(df.isnull().sum(), columns=['Missing Values'])
missing_values


# In[8]:


# Duplicated data
df.duplicated().sum()


# In[9]:


#Declare Feature Vector and Target Variable
X = df.drop('Outcome', axis=1)
y = df[['Outcome']]
y = y.values.ravel()


# In[10]:


# Check the shape of X and y
print ('X:', X.shape,'\ny:', y.shape)


# In[11]:


# Train Test Split (test_size=0.25)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)


# In[12]:


# Create a MinMaxScaler instance and fit it to the training data
scaler = MinMaxScaler() # saga solver requires features to be scaled for model conversion
X_train = scaler.fit_transform(X_train)

# Transform the test data using the same scaler
X_test = scaler.transform(X_test)

# Create a LogisticRegression instance and fit it to the scaled training data
logreg = LogisticRegression() # (default penalty = "l2") / (default solver = "lbfgs")
logreg.fit(X_train, y_train)

# Predict using the scaled test data
y_pred = logreg.predict(X_test)

print('Accuracy of logistic regression classifier on train set: {:.2f}'.format(logreg.score(X_train, y_train)))
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
print('Precision of logistic regression classifier on test set: {:.2f}'.format(precision_score(y_test, y_pred)))


# In[13]:


# default solver = 'lbfgs'
logreg = LogisticRegression(penalty='none')
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

print('Accuracy of logistic regression classifier on train set: {:.2f}'.format(logreg.score(X_train, y_train)))
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
print('Precision of logistic regression classifier on test set: {:.2f}'.format(precision_score(y_test, y_pred)))


# In[14]:


# penalty = “None”
clf = [
    LogisticRegression(solver='newton-cg',penalty='none'),
    LogisticRegression(solver='sag',penalty='none'),
    LogisticRegression(solver='saga',penalty='none')
    ]
clf_columns = []
clf_compare = pd.DataFrame(columns = clf_columns)

row_index = 0
for alg in clf:
        
    predicted = alg.fit(X_train, y_train).predict(X_test)
    fp, tp, th = roc_curve(y_test, predicted)
    clf_name = alg.__class__.__name__
    clf_compare.loc[row_index, 'Train Accuracy'] = round(alg.score(X_train, y_train), 5)
    clf_compare.loc[row_index, 'Test Accuracy'] = round(alg.score(X_test, y_test), 5)
    clf_compare.loc[row_index, 'Precision'] = round(precision_score(y_test, predicted),5)
    clf_compare.loc[row_index, 'Recall'] = round(recall_score(y_test, predicted),5)
    clf_compare.loc[row_index, 'AUC'] = round(auc(fp, tp),5)

    row_index+=1
    
clf_compare.sort_values(by = ['Test Accuracy'], ascending = False, inplace = True)    
clf_compare


# In[15]:


# penalty = 'l2'
clf = [
    LogisticRegression(solver='newton-cg',penalty='l2'),
    LogisticRegression(solver='sag',penalty='l2'),
    LogisticRegression(solver='saga',penalty='l2'),
    ]
clf_columns = []
clf_compare = pd.DataFrame(columns = clf_columns)

row_index = 0
for alg in clf:
        
    predicted = alg.fit(X_train, y_train).predict(X_test)
    fp, tp, th = roc_curve(y_test, predicted)
    clf_name = alg.__class__.__name__
    clf_compare.loc[row_index, 'Train Accuracy'] = round(alg.score(X_train, y_train), 5)
    clf_compare.loc[row_index, 'Test Accuracy'] = round(alg.score(X_test, y_test), 5)
    clf_compare.loc[row_index, 'Precision '] = round(precision_score(y_test, predicted),5)
    clf_compare.loc[row_index, 'Recall'] = round(recall_score(y_test, predicted),5)
    clf_compare.loc[row_index, 'AUC'] = round(auc(fp, tp),5)

    row_index+=1
    
clf_compare.sort_values(by = ['Test Accuracy'], ascending = False, inplace = True)    
clf_compare


# In[16]:


# solver = “liblinear” with penalty = “l1”
logreg = LogisticRegression(solver='liblinear', penalty='l1')
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

print('Accuracy of logistic regression classifier on train set: {:.2f}'.format(logreg.score(X_train, y_train)))
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
print('Precision of logistic regression classifier on test set: {:.2f}'.format(precision_score(y_test, y_pred)))


# In[17]:


# solver = “liblinear” with penalty = “l2”
logreg = LogisticRegression(solver='liblinear', penalty='l2')
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

print('Accuracy of logistic regression classifier on train set: {:.2f}'.format(logreg.score(X_train, y_train)))
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
print('Precision of logistic regression classifier on test set: {:.2f}'.format(precision_score(y_test, y_pred)))


# In[18]:


# Comparing “max_iter” parameter
clf = [
    LogisticRegression(solver='newton-cg',penalty='none',max_iter=100),
    LogisticRegression(solver='lbfgs',penalty='none',max_iter=200),
    LogisticRegression(solver='sag',penalty='none',max_iter=500),
    LogisticRegression(solver='saga',penalty='none',max_iter=1000)
    ]
clf_columns = []
clf_compare = pd.DataFrame(columns = clf_columns)

row_index = 0
for alg in clf:
        
    predicted = alg.fit(X_train, y_train).predict(X_test)
    fp, tp, th = roc_curve(y_test, predicted)
    clf_name = alg.__class__.__name__
    clf_compare.loc[row_index, 'Train Accuracy'] = round(alg.score(X_train, y_train), 5)
    clf_compare.loc[row_index, 'Test Accuracy'] = round(alg.score(X_test, y_test), 5)
    clf_compare.loc[row_index, 'Precision'] = round(precision_score(y_test, predicted),5)
    clf_compare.loc[row_index, 'Recall'] = round(recall_score(y_test, predicted),5)
    clf_compare.loc[row_index, 'AUC'] = round(auc(fp, tp),5)

    row_index+=1
    
clf_compare.sort_values(by = ['Test Accuracy'], ascending = False, inplace = True)    
clf_compare


# In[19]:


# Comparing “C” parameter
clf = [
    LogisticRegression(solver='newton-cg',penalty='l2', C=1),
    LogisticRegression(solver='lbfgs',penalty='l2',C=2.5),
    LogisticRegression(solver='sag',penalty='l2',C=5),
    LogisticRegression(solver='saga',penalty='l2',C=10)
    ]
clf_columns = []
clf_compare = pd.DataFrame(columns = clf_columns)

row_index = 0
for alg in clf:
        
    predicted = alg.fit(X_train, y_train).predict(X_test)
    fp, tp, th = roc_curve(y_test, predicted)
    clf_name = alg.__class__.__name__
    clf_compare.loc[row_index, 'Train Accuracy'] = round(alg.score(X_train, y_train), 5)
    clf_compare.loc[row_index, 'Test Accuracy'] = round(alg.score(X_test, y_test), 5)
    clf_compare.loc[row_index, 'Precission'] = round(precision_score(y_test, predicted),5)
    clf_compare.loc[row_index, 'Recall'] = round(recall_score(y_test, predicted),5)
    clf_compare.loc[row_index, 'AUC'] = round(auc(fp, tp),5)

    row_index+=1
    
clf_compare.sort_values(by = ['Test Accuracy'], ascending = False, inplace = True)    
clf_compare


# In[20]:


# Comparing “tol” parameter
clf = [
    LogisticRegression(solver='newton-cg',penalty='l2', tol=0.0001),
    LogisticRegression(solver='lbfgs',penalty='l2', tol=0.1),
    LogisticRegression(solver='sag',penalty='l2', tol=1),
    LogisticRegression(solver='saga',penalty='l2', tol=10)
    ]
clf_columns = []
clf_compare = pd.DataFrame(columns = clf_columns)

row_index = 0
for alg in clf:
        
    predicted = alg.fit(X_train, y_train).predict(X_test)
    fp, tp, th = roc_curve(y_test, predicted)
    clf_name = alg.__class__.__name__
    clf_compare.loc[row_index, 'Train Accuracy'] = round(alg.score(X_train, y_train), 5)
    clf_compare.loc[row_index, 'Test Accuracy'] = round(alg.score(X_test, y_test), 5)
    clf_compare.loc[row_index, 'Precission'] = round(precision_score(y_test, predicted),5)
    clf_compare.loc[row_index, 'Recall'] = round(recall_score(y_test, predicted),5)
    clf_compare.loc[row_index, 'AUC'] = round(auc(fp, tp),5)

    row_index+=1
    
clf_compare.sort_values(by = ['Test Accuracy'], ascending = False, inplace = True)    
clf_compare


# In[21]:


# Comparing “fit_intercept” parameter
clf = [
    LogisticRegression(solver='liblinear', penalty='l1', fit_intercept=False),
    LogisticRegression(solver='liblinear', penalty='l1', fit_intercept=True),
    ]
clf_columns = []
clf_compare = pd.DataFrame(columns = clf_columns)

row_index = 0
for alg in clf:
        
    predicted = alg.fit(X_train, y_train).predict(X_test)
    fp, tp, th = roc_curve(y_test, predicted)
    clf_name = alg.__class__.__name__
    clf_compare.loc[row_index, 'Train Accuracy'] = round(alg.score(X_train, y_train), 5)
    clf_compare.loc[row_index, 'Test Accuracy'] = round(alg.score(X_test, y_test), 5)
    clf_compare.loc[row_index, 'Precission'] = round(precision_score(y_test, predicted),5)
    clf_compare.loc[row_index, 'Recall'] = round(recall_score(y_test, predicted),5)
    clf_compare.loc[row_index, 'AUC'] = round(auc(fp, tp),5)

    row_index+=1
    
clf_compare.sort_values(by = ['Test Accuracy'], ascending = False, inplace = True)    
clf_compare


# In[22]:


# Comparing “intercept_scaling” parameter
clf = [
    LogisticRegression(solver='liblinear', penalty='l1', fit_intercept=True, intercept_scaling=0.01),
    LogisticRegression(solver='liblinear', penalty='l1', fit_intercept=True, intercept_scaling=0.1),
    LogisticRegression(solver='liblinear', penalty='l1', fit_intercept=True, intercept_scaling=0.2),
    LogisticRegression(solver='liblinear', penalty='l1', fit_intercept=True, intercept_scaling=0.5),
    LogisticRegression(solver='liblinear', penalty='l1', fit_intercept=True, intercept_scaling=1),
    LogisticRegression(solver='liblinear', penalty='l1', fit_intercept=True, intercept_scaling=2),
    LogisticRegression(solver='liblinear', penalty='l1', fit_intercept=True, intercept_scaling=5),
    LogisticRegression(solver='liblinear', penalty='l1', fit_intercept=True, intercept_scaling=10),
    ]
clf_columns = []
clf_compare = pd.DataFrame(columns = clf_columns)

row_index = 0
for alg in clf:
        
    predicted = alg.fit(X_train, y_train).predict(X_test)
    fp, tp, th = roc_curve(y_test, predicted)
    clf_name = alg.__class__.__name__
    clf_compare.loc[row_index, 'Train Accuracy'] = round(alg.score(X_train, y_train), 5)
    clf_compare.loc[row_index, 'Test Accuracy'] = round(alg.score(X_test, y_test), 5)
    clf_compare.loc[row_index, 'Precission'] = round(precision_score(y_test, predicted),5)
    clf_compare.loc[row_index, 'Recall'] = round(recall_score(y_test, predicted),5)
    clf_compare.loc[row_index, 'AUC'] = round(auc(fp, tp),5)

    row_index+=1
    
clf_compare.sort_values(by = ['Test Accuracy'], ascending = False, inplace = True)    
clf_compare


# In[23]:


# Comparing “class_weight” parameter
clf = [
    LogisticRegression(solver='liblinear', penalty='l1', fit_intercept=True,
                       intercept_scaling=0.2, class_weight='balanced'),
    LogisticRegression(solver='liblinear', penalty='l1', fit_intercept=True,
                       intercept_scaling=0.2, class_weight=None),
    ]
clf_columns = []
clf_compare = pd.DataFrame(columns = clf_columns)

row_index = 0
for alg in clf:
        
    predicted = alg.fit(X_train, y_train).predict(X_test)
    fp, tp, th = roc_curve(y_test, predicted)
    clf_name = alg.__class__.__name__
    clf_compare.loc[row_index, 'Train Accuracy'] = round(alg.score(X_train, y_train), 5)
    clf_compare.loc[row_index, 'Test Accuracy'] = round(alg.score(X_test, y_test), 5)
    clf_compare.loc[row_index, 'Precission'] = round(precision_score(y_test, predicted),5)
    clf_compare.loc[row_index, 'Recall'] = round(recall_score(y_test, predicted),5)
    clf_compare.loc[row_index, 'AUC'] = round(auc(fp, tp),5)

    row_index+=1
    
clf_compare.sort_values(by = ['Test Accuracy'], ascending = False, inplace = True)    
clf_compare


# In[24]:


# Comparing “random_state” parameter
clf = [
    LogisticRegression(solver='liblinear', penalty='l1', fit_intercept=True,
                       intercept_scaling=0.2, class_weight=None, random_state=None),
    LogisticRegression(solver='liblinear', penalty='l1', fit_intercept=True,
                       intercept_scaling=0.2, class_weight=None, random_state=0),
    LogisticRegression(solver='liblinear', penalty='l1', fit_intercept=True,
                       intercept_scaling=0.2, class_weight=None, random_state=100)
    ]
clf_columns = []
clf_compare = pd.DataFrame(columns = clf_columns)

row_index = 0
for alg in clf:
        
    predicted = alg.fit(X_train, y_train).predict(X_test)
    fp, tp, th = roc_curve(y_test, predicted)
    clf_name = alg.__class__.__name__
    clf_compare.loc[row_index, 'Train Accuracy'] = round(alg.score(X_train, y_train), 5)
    clf_compare.loc[row_index, 'Test Accuracy'] = round(alg.score(X_test, y_test), 5)
    clf_compare.loc[row_index, 'Precission'] = round(precision_score(y_test, predicted),5)
    clf_compare.loc[row_index, 'Recall'] = round(recall_score(y_test, predicted),5)
    clf_compare.loc[row_index, 'AUC'] = round(auc(fp, tp),5)

    row_index+=1
    
clf_compare.sort_values(by = ['Test Accuracy'], ascending = False, inplace = True)    
clf_compare


# In[25]:


# Comparing “multi_class” parameter
clf = [
    LogisticRegression(solver='liblinear', penalty='l1', fit_intercept=True,
                       intercept_scaling=0.2, class_weight=None, multi_class='auto'),
    LogisticRegression(solver='liblinear', penalty='l1', fit_intercept=True,
                       intercept_scaling=0.2, class_weight=None, multi_class= 'ovr' ),
    LogisticRegression(solver='lbfgs', penalty='l2', fit_intercept=True,
                       intercept_scaling=0.2, class_weight=None, multi_class= 'auto'),
    LogisticRegression(solver='lbfgs', penalty='l2', fit_intercept=True,
                       intercept_scaling=0.2, class_weight=None, multi_class= 'ovr'),
    LogisticRegression(solver='lbfgs', penalty='l2', fit_intercept=True,
                       intercept_scaling=0.2, class_weight=None, multi_class= 'multinomial')
    ]
clf_columns = []
clf_compare = pd.DataFrame(columns = clf_columns)    

row_index = 0
for alg in clf:
        
    predicted = alg.fit(X_train, y_train).predict(X_test)
    fp, tp, th = roc_curve(y_test, predicted)
    clf_name = alg.__class__.__name__
    clf_compare.loc[row_index, 'Train Accuracy'] = round(alg.score(X_train, y_train), 5)
    clf_compare.loc[row_index, 'Test Accuracy'] = round(alg.score(X_test, y_test), 5)
    clf_compare.loc[row_index, 'Precission'] = round(precision_score(y_test, predicted),5)
    clf_compare.loc[row_index, 'Recall'] = round(recall_score(y_test, predicted),5)
    clf_compare.loc[row_index, 'AUC'] = round(auc(fp, tp),5)

    row_index+=1
    
clf_compare.sort_values(by = ['Test Accuracy'], ascending = False, inplace = True)    
clf_compare


# In[26]:


# Comparing “verbose” parameter
clf = [
    LogisticRegression(solver='liblinear', penalty='l1', fit_intercept=True,
                       intercept_scaling=0.2, class_weight=None, verbose =0.1),
    LogisticRegression(solver='liblinear', penalty='l1', fit_intercept=True,
                       intercept_scaling=0.2, class_weight=None, verbose =0),
    LogisticRegression(solver='liblinear', penalty='l1', fit_intercept=True,
                       intercept_scaling=0.2, class_weight=None, verbose =1),
    LogisticRegression(solver='liblinear', penalty='l1', fit_intercept=True,
                       intercept_scaling=0.2, class_weight=None, verbose =10),
    ]
clf_columns = []
clf_compare = pd.DataFrame(columns = clf_columns)    

row_index = 0
for alg in clf:
        
    predicted = alg.fit(X_train, y_train).predict(X_test)
    fp, tp, th = roc_curve(y_test, predicted)
    clf_name = alg.__class__.__name__
    clf_compare.loc[row_index, 'Train Accuracy'] = round(alg.score(X_train, y_train), 5)
    clf_compare.loc[row_index, 'Test Accuracy'] = round(alg.score(X_test, y_test), 5)
    clf_compare.loc[row_index, 'Precission'] = round(precision_score(y_test, predicted),5)
    clf_compare.loc[row_index, 'Recall'] = round(recall_score(y_test, predicted),5)
    clf_compare.loc[row_index, 'AUC'] = round(auc(fp, tp),5)

    row_index+=1
    
clf_compare.sort_values(by = ['Test Accuracy'], ascending = False, inplace = True)    
clf_compare


# In[27]:


# Comparing “warm_start” parameter
clf = [
    LogisticRegression(solver='lbfgs', penalty='none', fit_intercept=True,
                       intercept_scaling=0.2, class_weight=None, warm_start =True),
    LogisticRegression(solver='lbfgs', penalty='none', fit_intercept=True,
                       intercept_scaling=0.2, class_weight=None, warm_start =False),
    LogisticRegression(solver='newton-cg', penalty='none', fit_intercept=True,
                       intercept_scaling=0.2, class_weight=None, warm_start =True),
    LogisticRegression(solver='newton-cg', penalty='none', fit_intercept=True,
                       intercept_scaling=0.2, class_weight=None, warm_start =False),
    LogisticRegression(solver='sag', penalty='none', fit_intercept=True,
                       intercept_scaling=0.2, class_weight=None, warm_start =True),
    LogisticRegression(solver='sag', penalty='none', fit_intercept=True,
                       intercept_scaling=0.2, class_weight=None, warm_start =False),
    LogisticRegression(solver='saga', penalty='none', fit_intercept=True,
                       intercept_scaling=0.2, class_weight=None, warm_start =True),
    LogisticRegression(solver='saga', penalty='none', fit_intercept=True,
                       intercept_scaling=0.2, class_weight=None, warm_start =False),
    ]
clf_columns = []
clf_compare = pd.DataFrame(columns = clf_columns)    

row_index = 0
for alg in clf:
        
    predicted = alg.fit(X_train, y_train).predict(X_test)
    fp, tp, th = roc_curve(y_test, predicted)
    clf_name = alg.__class__.__name__
    clf_compare.loc[row_index, 'Train Accuracy'] = round(alg.score(X_train, y_train), 5)
    clf_compare.loc[row_index, 'Test Accuracy'] = round(alg.score(X_test, y_test), 5)
    clf_compare.loc[row_index, 'Precission'] = round(precision_score(y_test, predicted),5)
    clf_compare.loc[row_index, 'Recall'] = round(recall_score(y_test, predicted),5)
    clf_compare.loc[row_index, 'AUC'] = round(auc(fp, tp),5)

    row_index+=1
    
clf_compare.sort_values(by = ['Test Accuracy'], ascending = False, inplace = True)    
clf_compare


# In[28]:


# Comparing “l1_ratio” parameter
clf = [
    LogisticRegression(solver='saga', penalty='elasticnet', fit_intercept=True,
                       intercept_scaling=0.2, class_weight=None, l1_ratio=0),
    LogisticRegression(solver='saga', penalty='elasticnet', fit_intercept=True,
                       intercept_scaling=0.2, class_weight=None, l1_ratio=1),
    LogisticRegression(solver='saga', penalty='elasticnet', fit_intercept=True,
                       intercept_scaling=0.2, class_weight=None, l1_ratio=0.5),
    ]
clf_columns = []
clf_compare = pd.DataFrame(columns = clf_columns)    

row_index = 0
for alg in clf:
        
    predicted = alg.fit(X_train, y_train).predict(X_test)
    fp, tp, th = roc_curve(y_test, predicted)
    clf_name = alg.__class__.__name__
    clf_compare.loc[row_index, 'Train Accuracy'] = round(alg.score(X_train, y_train), 5)
    clf_compare.loc[row_index, 'Test Accuracy'] = round(alg.score(X_test, y_test), 5)
    clf_compare.loc[row_index, 'Precission'] = round(precision_score(y_test, predicted),5)
    clf_compare.loc[row_index, 'Recall'] = round(recall_score(y_test, predicted),5)
    clf_compare.loc[row_index, 'AUC'] = round(auc(fp, tp),5)

    row_index+=1
    
clf_compare.sort_values(by = ['Test Accuracy'], ascending = False, inplace = True)    
clf_compare


# #### Model 1: Default hyperparameters

# In[29]:


logreg = LogisticRegression() # default hyperparameters was seted for modeling
logreg.fit(X_train, y_train)

# Make predictions on the test set
y_pred = logreg.predict(X_test)

print('Accuracy of logistic regression classifier on train set: {:.2f}'.format(logreg.score(X_train, y_train)))
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))


# In[30]:


# Confusion Matrix
ConfusionMatrixDisplay.from_estimator(logreg, X_train, y_train, colorbar=False, cmap='YlGnBu')
plt.title('Model 1-Confusion Matrix (train)')
plt.grid(False)


# In[31]:


# Confusion Matrix
ConfusionMatrixDisplay.from_estimator(logreg, X_test, y_test, colorbar=False, cmap='YlGnBu')
plt.title('Model 1-Confusion Matrix (test)')
plt.grid(False) 


# In[32]:


print(classification_report(y_test, y_pred))


# In[33]:


# Calculate Performance Metrics
def metrics_calculator(y_test, y_pred, model_name):
    '''
    This function calculates all desired performance metrics for a given model.
    '''
    result = pd.DataFrame(data=[accuracy_score(y_test, y_pred),
                                precision_score(y_test, y_pred, average='macro'),
                                recall_score(y_test, y_pred, average='macro'),
                                f1_score(y_test, y_pred, average='macro')],
                          index=['Accuracy','Precision','Recall','F1-score'],
                          columns = [model_name])
    return result


# In[34]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# In[35]:


Model1_result = metrics_calculator(y_test, y_pred, 'Model 1 (Default)')
Model1_result


# In[36]:


# Precision-Recall Curve (PRC)
sns.set_style('darkgrid')
fig, ax = plt.subplots(figsize=(6, 5), dpi=80)
PrecisionRecallDisplay.from_estimator(logreg, X_test, y_test, ax=ax, name='Model 1', color='#6198c7')
plt.title('Precision-Recall Curve (Model 1)')
plt.show()


# In[37]:


# ROC Curve
fig, ax = plt.subplots(figsize=(6, 5), dpi=80)
RocCurveDisplay.from_estimator(logreg, X_test, y_test, ax=ax, name='Model 1', color='#6198c7')
plt.title('ROC Curve (Model 1)')
plt.show()


# #### Model 2: Best hyperparameters for “liblinear” (Manually tuned)

# In[38]:


# best parameters for 'liblinear' solver
logreg = LogisticRegression(solver='liblinear', penalty='l1', C=10,
                            fit_intercept= True, intercept_scaling= 0.2,
                            class_weight=None)
logreg.fit(X_train, y_train)

# Make predictions on the test set
y_pred = logreg.predict(X_test)

print('Accuracy of logistic regression classifier on train set: {:.2f}'.format(logreg.score(X_train, y_train)))
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))


# In[39]:


# Confusion Matrix
ConfusionMatrixDisplay.from_estimator(logreg, X_train, y_train, colorbar=False, cmap='YlGnBu')
plt.title('Model 2-Confusion Matrix (train)')
plt.grid(False)


# In[40]:


# Confusion Matrix
ConfusionMatrixDisplay.from_estimator(logreg, X_test, y_test, colorbar=False, cmap='YlGnBu')
plt.title('Model 2-Confusion Matrix (test)')
plt.grid(False) 


# In[41]:


print(classification_report(y_test, y_pred))


# In[42]:


Model2_result = metrics_calculator(y_test, y_pred, 'Model 2 (liblinear)')
Model2_result


# In[43]:


# Precision-Recall Curve (PRC)
fig, ax = plt.subplots(figsize=(6, 5), dpi=80)
PrecisionRecallDisplay.from_estimator(logreg, X_test, y_test, ax=ax, name='Model 2', color='#97c761')
plt.title('Precision-Recall Curve (Model 2)')
plt.show()


# In[44]:


# ROC Curve
fig, ax = plt.subplots(figsize=(6, 5), dpi=80)
RocCurveDisplay.from_estimator(logreg, X_test, y_test, ax=ax, name='Model 2', color='#97c761')
plt.title('ROC Curve (Model 2)')
plt.show()


# #### Model 3: Best hyperparameters for “lbfgs” (Manually tuned)

# In[45]:


logreg = LogisticRegression(solver='lbfgs', penalty='none', C=1, fit_intercept= True,
                            intercept_scaling= 0.2, class_weight=None, multi_class='multinomial') 
logreg.fit(X_train, y_train)

# Make predictions on the test set
y_pred = logreg.predict(X_test)

print('Accuracy of logistic regression classifier on train set: {:.2f}'.format(logreg.score(X_train, y_train)))
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))


# In[46]:


# Confusion Matrix
ConfusionMatrixDisplay.from_estimator(logreg, X_train, y_train, colorbar=False, cmap='YlGnBu')
plt.title('Model 3-Confusion Matrix (train)')
plt.grid(False)


# In[47]:


# Confusion Matrix
ConfusionMatrixDisplay.from_estimator(logreg, X_test, y_test, colorbar=False, cmap='YlGnBu')
plt.title('Model 3-Confusion Matrix (test)')
plt.grid(False) 


# In[48]:


print(classification_report(y_test, y_pred))


# In[49]:


Model3_result = metrics_calculator(y_test, y_pred, 'Model 3 (lbfgs)')
Model3_result


# In[50]:


# Precision-Recall Curve (PRC)
fig, ax = plt.subplots(figsize=(6, 5), dpi=80)
PrecisionRecallDisplay.from_estimator(logreg, X_test, y_test, ax=ax, name='Model 3', color='#c76194')
plt.title('Precision-Recall Curve (Model 3)')
plt.show()


# In[51]:


# ROC Curve
fig, ax = plt.subplots(figsize=(6, 5), dpi=80)
RocCurveDisplay.from_estimator(logreg, X_test, y_test, ax=ax, name='Model 3', color='#c76194')
plt.title('ROC Curve (Model 3)')
plt.show()


# #### Model 4: Best hyperparameters for “newton-cg” (Manually tuned)

# In[52]:


logreg = LogisticRegression(solver='newton-cg', penalty='none', C=1, fit_intercept= True,
                            intercept_scaling= 0.2, class_weight=None) 
logreg.fit(X_train, y_train)

# Make predictions on the test set
y_pred = logreg.predict(X_test)

print('Accuracy of logistic regression classifier on train set: {:.2f}'.format(logreg.score(X_train, y_train)))
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))


# In[53]:


# Confusion Matrix
ConfusionMatrixDisplay.from_estimator(logreg, X_train, y_train, colorbar=False, cmap='YlGnBu')
plt.title('Model 4-Confusion Matrix (train)')
plt.grid(False)


# In[54]:


# Confusion Matrix
ConfusionMatrixDisplay.from_estimator(logreg, X_test, y_test, colorbar=False, cmap='YlGnBu')
plt.title('Model 4-Confusion Matrix (test)')
plt.grid(False) 


# In[55]:


print(classification_report(y_test, y_pred))


# In[56]:


Model4_result = metrics_calculator(y_test, y_pred, 'Model 4 (newton-cg)')
Model4_result


# In[57]:


# Precision-Recall Curve (PRC)
fig, ax = plt.subplots(figsize=(6, 5), dpi=80)
PrecisionRecallDisplay.from_estimator(logreg, X_test, y_test, ax=ax, name='Model 4', color='#d49b55')
plt.title('Precision-Recall Curve (Model 4)')
plt.show()


# In[58]:


# ROC Curve
fig, ax = plt.subplots(figsize=(6, 5), dpi=80)
RocCurveDisplay.from_estimator(logreg, X_test, y_test, ax=ax, name='Model 4', color='#d49b55')
plt.title('ROC Curve (Model 4)')
plt.show()


# In[59]:


# Conclusion
pd.concat([Model1_result, Model2_result, Model3_result, Model4_result], axis=1)

