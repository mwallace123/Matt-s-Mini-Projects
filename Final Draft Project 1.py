#!/usr/bin/env python
# coding: utf-8

# # Introduction

# The following project investigates tabulated data where each row represents a single customer and the columns represent the customer's evaluation based on various metrics outlined below.

# In[1276]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn
import seaborn as sns
import warnings
warnings.filterwarnings("ignore") # this will take away the red dialog boxes in the output terminal


# In[1277]:


file = pd.read_csv('project1data')


# In[1278]:


file


# ### Exploratory Data Analysis

# #### X1: My Order Was Delivered On Time

# In[1279]:


file['X1'].describe()


# In[1280]:


plt.hist(file['X1'])
plt.title('My Order Was Delivered On Time')
plt.show()


# The majority of orders were in the 4 and 5 category indicating great satisfaction with time delivery

# #### X2: Contents of my order was as I expected

# In[1281]:


file['X2'].describe()


# In[1282]:


plt.hist(file['X2'])
plt.title('The Contents Of My Order Was As I Expected')
plt.show()


# The majority of the orders were in the 1, 2 and 3 category indicating dissatisfaction with the contents of the order.

# #### X3: I ordered everything I wanted to order

# In[1283]:


file['X3'].describe()


# In[1284]:


plt.hist(file['X3'])
plt.title('I ordered Everything I Wanted To Order')
plt.show()


# The distribution looks like a normal curve scewed to the right indicating moderate satisfaction with the customers opportunity to order everything they want to order

# #### X4: I paid a good price for my order

# In[1285]:


file['X4'].describe()


# In[1286]:


plt.hist(file['X4'])
plt.title('I Paid A Good Price For My Order')
plt.show()


# Most customers were not 100% satisfied with the price but most didn't see price as disputable.

# #### X5: I am satisfied with my courier.

# In[1287]:


file['X5'].describe()


# In[1288]:


plt.hist(file['X5'])
plt.title('I Am Satisfied With My Courier')
plt.show()


# The majority of orders were in the 3 and higher category indicating moderate satisfaction with the courier.

# #### X6: The app makes the ordering easy for me.

# In[1289]:


file['X6'].describe()


# In[1290]:


plt.hist(file['X6'])
plt.title('The App Makes The Ordering Easy For Me')
plt.show()


# The majority of the orders are in the 4 and 5 category for using the ordering app.  Many people like using the app and may result in a positive influence in overall satisfaction.

# In[1291]:


Averages = {'X1': np.mean(file['X1']), 
            'X2': np.mean(file['X2']), 
            'X3': np.mean(file['X3']), 
            'X4': np.mean(file['X4']),
            'X5': np.mean(file['X5']),
            'X6': np.mean(file['X6'])}
Averages = pd.DataFrame(Averages, index = [0])
Averages

index = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6']
values = [np.mean(file['X1']), np.mean(file['X2']), np.mean(file['X3']), np.mean(file['X4']), np.mean(file['X5']), np.mean(file['X6'])]


# In[1292]:


Averages


# In[1293]:


values = pd.DataFrame(values, index = index) # pivot table for plotting


# In[1294]:


values


# In[1295]:


plt.scatter(x = index, y=values, s = 200)
plt.xlabel('Attribute')
plt.ylabel('Average For Each Attribute')
plt.title('Average Satisfaction Rate For Each Attribute')
plt.show()


# In Summary, Customers were happiest with their orders being delivered on time and using the app.  Customers were not as happy with the way the contents of the order were not as expected.  Customers were moderately happy with the price and their courier but not as happy with opportunity to order everything they want.

# #### Y: Target Attribute; overall satisfaction

# In[1296]:


file["Y"].value_counts(normalize=True).plot.bar(title='Target Attribute: Overall Satisfaction')
plt.show()


# More customers were happy with their orders but there are also a substantial amount of unhappy customers.

# ### Correlations Between Each Variable And The Target Variable

# In[1297]:


import seaborn as sns


# In[1298]:


_= sns.lmplot(x = 'X1', y = 'Y', data = file, fit_reg = False)
plt.title('Scatter plot between X1 and Y')
plt.show()


# In[1299]:


_= sns.lmplot(x = 'X2', y = 'Y', data = file, fit_reg = False)
plt.title('Scatter plot between X2 and Y')
plt.show()


# In[1300]:


_= sns.lmplot(x = 'X3', y = 'Y', data = file, fit_reg = False)
plt.title('Scatter plot between X3 and Y')
plt.show()


# In[1301]:


_= sns.lmplot(x = 'X4', y = 'Y', data = file, fit_reg = False)
plt.title('Scatter plot between X4 and Y')
plt.show()


# In[1302]:


_= sns.lmplot(x = 'X5', y = 'Y', data = file, fit_reg = False)
plt.title('Scatter plot between X5 and Y')
plt.show()


# In[1303]:


_= sns.lmplot(x = 'X6', y = 'Y', data = file, fit_reg = False)
plt.title('Scatter plot between X1 and Y')
plt.show()


# ##### There is no observable correlation between any of the single variables and the target variable.

# In[1304]:


plt.figure(figsize = (20, 20))
file.corr().abs()
z = file.corr().abs()
sns.heatmap(data=z, annot=True)
plt.title('Correlation Heatmap Between All Variables In The Dataset')
plt.show()


# ###### As confirmed by the correlation heatmap above, the highest correlation coefficient is 0.28 between X1 and Y and the lowest correlation at 0.024 between X2 and Y.

# # Machine Learning Algorithms

# The below cells are the creations of four algorithms:
# 1. Logistic Regression
# 2. Random Forest
# 3. Decision Tree
# 4. Gradient Boosted Decision Tree

# In[1305]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

def run_regression_accuracy(X_train, y_train, X_test, y_test):   
    paramgrid = {'C': np.logspace(-8, 5, 15)}
    grid_search = GridSearchCV(LogisticRegression(), paramgrid)
    
    grid_search.fit(X_train, y_train)
    logreg = grid_search.best_estimator_
    y_pred = logreg.predict(X_test)
    print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
    cm = confusion_matrix(y_test, y_pred)
    print('\nConfusion matrix: \n',cm)

    print('\nClassification report: \n',classification_report(y_test, y_pred))

    logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
    fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])

    plt.figure()
    plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('LOGISTIC REGRESSION RESULTS.png')
    plt.show()
  
    return logreg


# In[1306]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

def run_random_forest_accuracy(X_train, y_train, X_test, y_test):   
    paramgrid = {'max_depth': list(range(1, 20, 2)), 'n_estimators': list(range(1, 200, 20))}
    grid_search = RandomizedSearchCV(RandomForestClassifier(random_state=1), paramgrid) # Could do a RandomizedSearchCV to save computing.
    
    grid_search.fit(X_train, y_train)
    logreg = grid_search.best_estimator_
    y_pred = logreg.predict(X_test)
    print('Accuracy of random forest classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
    cm = confusion_matrix(y_test, y_pred)
    print('\nConfusion matrix: \n',cm)

    print('\nClassification report: \n',classification_report(y_test, y_pred))

    logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
    fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])

    plt.figure()
    plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('RANDOM FOREST RESULTS.png')
    plt.show()
  
    return logreg


# In[1307]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV

def run_decision_tree_accuracy(X_train, y_train, X_test, y_test):   
    paramgrid = {'max_depth': list(range(1, 20, 2))
                }
    grid_search = RandomizedSearchCV(DecisionTreeClassifier(), paramgrid, cv=10) # Could do a RandomizedSearchCV to save computing.
    
    grid_search.fit(X_train, y_train)
    logreg = grid_search.best_estimator_
    y_pred = logreg.predict(X_test)
    print('Accuracy of decision tree on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
    cm = confusion_matrix(y_test, y_pred)
    print('\nConfusion matrix: \n',cm)

    print('\nClassification report: \n',classification_report(y_test, y_pred))

    logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
    fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])

    plt.figure()
    plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('RANDOM FOREST RESULTS.png')
    plt.show()
  
    return logreg


# In[1308]:


import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV

def run_gradient_boosted_decision_tree_accuracy(X_train, y_train, X_test, y_test):   
    paramgrid = {'num_leaves': np.arange(20, 200, 5),
    'learning_rate': np.logspace(-3, 0, 100),
    'n_estimators': np.arange(50, 500, 10),
    'min_child_samples': np.arange(5, 50, 5)
                }
    grid_search = RandomizedSearchCV(lgb.LGBMClassifier(), paramgrid, cv=5, n_iter=100, scoring='accuracy') # Could do a RandomizedSearchCV to save computing.
    
    grid_search.fit(X_train, y_train)
    logreg = grid_search.best_estimator_
    y_pred = logreg.predict(X_test)
    print('Accuracy of boosted decision tree on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
    cm = confusion_matrix(y_test, y_pred)
    print('\nConfusion matrix: \n',cm)

    print('\nClassification report: \n',classification_report(y_test, y_pred))

    logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
    fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])

    plt.figure()
    plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('RANDOM FOREST RESULTS.png')
    plt.show()
  
    return logreg


# In[1309]:


import shap


# Splitting the data up into the independent variables and the target variable

# In[1310]:


X = file.drop(['Y'], axis=1)
y = file['Y']


# Splitting the data up into training and test sets at 60% training data and 40% testing data

# In[1311]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)


# #### Application of models to training and test set

# In[1312]:


explainer = shap.LinearExplainer(run_regression_accuracy(X_train, y_train, X_test, y_test), X_train, feature_dependence="independent")
shap_values = explainer.shap_values(X_train)


# In[1313]:


shap.summary_plot(shap_values, X_train)


# In[1314]:


explainer = shap.TreeExplainer(run_random_forest_accuracy(X_train, y_train, X_test, y_test), X_train, feature_dependence="independent")
shap_values_random_forest = explainer.shap_values(X_train)


# In[1315]:


shap.summary_plot(shap_values, X_train, plot_type = "bar")


# In[1316]:


explainer = shap.TreeExplainer(run_decision_tree_accuracy(X_train, y_train, X_test, y_test), X_train, feature_dependence="independent")
shap_values = explainer.shap_values(X_train)


# In[1317]:


shap.summary_plot(shap_values, X_train, plot_type = "bar")


# In[1318]:


explainer = shap.TreeExplainer(run_gradient_boosted_decision_tree_accuracy(X_train, y_train, X_test, y_test), X_train, feature_dependence="independent")
shap_values = explainer.shap_values(X_train)


# In[1319]:


shap.summary_plot(shap_values, X_train, plot_type = "bar")


# Redefining the independent variables by taking a subset of the features via feature selection

# In[1320]:


X = file.drop(['Y', 'X2', 'X5'], axis=1)
y = file['Y']


# Rest of the Notebook Applies the algorithms through varying sizes of testing and training data

# In[1321]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)


# In[1322]:


explainer = shap.TreeExplainer(run_random_forest_accuracy(X_train, y_train, X_test, y_test), X_train, feature_dependence="independent")
shap_values = explainer.shap_values(X_train1)


# In[1323]:


shap.summary_plot(shap_values, X_train, plot_type = "bar")


# In[1324]:


# Create an introduction explaining the project and the goal of the project.
# Descriptions throughout the project as to what's been done.
# Tidy up the analysis and get rid of any empty cells.
# Conclusion.


# In[1325]:


explainer = shap.TreeExplainer(run_gradient_boosted_decision_tree_accuracy(X_train, y_train, X_test, y_test), X_train, feature_dependence="independent")
shap_values = explainer.shap_values(X_train1)


# In[1326]:


shap.summary_plot(shap_values, X_train, plot_type = "bar")


# In[1327]:


explainer = shap.TreeExplainer(run_decision_tree_accuracy(X_train, y_train, X_test, y_test), X_train, feature_dependence="independent")
shap_values = explainer.shap_values(X_train)


# In[1328]:


shap.summary_plot(shap_values, X_train, plot_type = "bar")


# In[1329]:


explainer = shap.LinearExplainer(run_regression_accuracy(X_train, y_train, X_test, y_test), X_train, feature_dependence="independent")
shap_values = explainer.shap_values(X_train)


# In[1330]:


shap.summary_plot(shap_values, X_train)


# In[1331]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)


# In[1332]:


explainer = shap.TreeExplainer(run_random_forest_accuracy(X_train, y_train, X_test, y_test), X_train, feature_dependence="independent")
shap_values = explainer.shap_values(X_train)


# In[1333]:


shap.summary_plot(shap_values, X_train)


# In[1334]:


explainer = shap.TreeExplainer(run_gradient_boosted_decision_tree_accuracy(X_train, y_train, X_test, y_test), X_train, feature_dependence="independent")
shap_values = explainer.shap_values(X_train)


# In[1335]:


shap.summary_plot(shap_values, X_train)


# In[1336]:


explainer = shap.TreeExplainer(run_decision_tree_accuracy(X_train, y_train, X_test, y_test), X_train, feature_dependence="independent")
shap_values = explainer.shap_values(X_train)


# In[1337]:


shap.summary_plot(shap_values, X_train)


# In[1338]:


explainer = shap.LinearExplainer(run_regression_accuracy(X_train, y_train, X_test, y_test), X_train, feature_dependence="independent")
shap_values = explainer.shap_values(X_train)


# In[1339]:


shap.summary_plot(shap_values, X_train)


# In[1340]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[1341]:


explainer = shap.TreeExplainer(run_random_forest_accuracy(X_train, y_train, X_test, y_test), X_train, feature_dependence="independent")
shap_values = explainer.shap_values(X_train)


# In[1342]:


shap.summary_plot(shap_values, X_train)


# In[1343]:


explainer = shap.TreeExplainer(run_gradient_boosted_decision_tree_accuracy(X_train, y_train, X_test, y_test), X_train, feature_dependence="independent")
shap_values = explainer.shap_values(X_train)


# In[1344]:


shap.summary_plot(shap_values, X_train)


# In[1345]:


explainer = shap.TreeExplainer(run_decision_tree_accuracy(X_train, y_train, X_test, y_test), X_train, feature_dependence="independent")
shap_values = explainer.shap_values(X_train)


# In[1346]:


shap.summary_plot(shap_values, X_train)


# In[1347]:


explainer = shap.LinearExplainer(run_regression_accuracy(X_train, y_train, X_test, y_test), X_train, feature_dependence="independent")
shap_values = explainer.shap_values(X_train)


# In[1348]:


shap.summary_plot(shap_values, X_train)


# In[1349]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# In[1350]:


explainer = shap.TreeExplainer(run_random_forest_accuracy(X_train, y_train, X_test, y_test), X_train, feature_dependence="independent")
shap_values = explainer.shap_values(X_train)


# In[1351]:


shap.summary_plot(shap_values, X_train)


# In[1352]:


explainer = shap.TreeExplainer(run_gradient_boosted_decision_tree_accuracy(X_train, y_train, X_test, y_test), X_train, feature_dependence="independent")
shap_values = explainer.shap_values(X_train)


# In[1353]:


shap.summary_plot(shap_values, X_train)


# In[1354]:


explainer = shap.TreeExplainer(run_decision_tree_accuracy(X_train, y_train, X_test, y_test), X_train, feature_dependence="independent")
shap_values = explainer.shap_values(X_train)


# In[1355]:


shap.summary_plot(shap_values, X_train)


# In[1356]:


explainer = shap.LinearExplainer(run_regression_accuracy(X_train, y_train, X_test, y_test), X_train, feature_dependence="independent")
shap_values = explainer.shap_values(X_train)


# In[1357]:


shap.summary_plot(shap_values, X_train)


# In[1358]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[1359]:


explainer = shap.TreeExplainer(run_random_forest_accuracy(X_train, y_train, X_test, y_test), X_train, feature_dependence="independent")
shap_values = explainer.shap_values(X_train)


# In[1360]:


shap.summary_plot(shap_values, X_train)


# In[1361]:


explainer = shap.TreeExplainer(run_decision_tree_accuracy(X_train, y_train, X_test, y_test), X_train, feature_dependence="independent")
shap_values = explainer.shap_values(X_train)


# In[1362]:


shap.summary_plot(shap_values, X_train)


# In[1363]:


explainer = shap.TreeExplainer(run_gradient_boosted_decision_tree_accuracy(X_train, y_train, X_test, y_test), X_train, feature_dependence="independent")
shap_values = explainer.shap_values(X_train)


# In[1364]:


shap.summary_plot(shap_values, X_train)


# In[1365]:


explainer = shap.LinearExplainer(run_regression_accuracy(X_train, y_train, X_test, y_test), X_train, feature_dependence="independent")
shap_values = explainer.shap_values(X_train)


# In[1366]:


shap.summary_plot(shap_values, X_train)


# In[1367]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)


# In[1368]:


explainer = shap.TreeExplainer(run_random_forest_accuracy(X_train, y_train, X_test, y_test), X_train, feature_dependence="independent")
shap_values = explainer.shap_values(X_train)


# In[1369]:


shap.summary_plot(shap_values, X_train)


# In[1370]:


explainer = shap.TreeExplainer(run_gradient_boosted_decision_tree_accuracy(X_train, y_train, X_test, y_test), X_train, feature_dependence="independent")
shap_values = explainer.shap_values(X_train)


# In[1371]:


shap.summary_plot(shap_values, X_train)


# In[1372]:


explainer = shap.TreeExplainer(run_decision_tree_accuracy(X_train, y_train, X_test, y_test), X_train, feature_dependence="independent")
shap_values = explainer.shap_values(X_train)


# In[1373]:


shap.summary_plot(shap_values, X_train)


# In[1374]:


explainer = shap.LinearExplainer(run_regression_accuracy(X_train, y_train, X_test, y_test), X_train, feature_dependence="independent")
shap_values = explainer.shap_values(X_train)


# In[1375]:


shap.summary_plot(shap_values, X_train)


# In[1376]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)


# In[1377]:


explainer = shap.TreeExplainer(run_random_forest_accuracy(X_train, y_train, X_test, y_test), X_train, feature_dependence="independent")
shap_values = explainer.shap_values(X_train)


# In[1378]:


shap.summary_plot(shap_values, X_train)


# In[1379]:


explainer = shap.TreeExplainer(run_gradient_boosted_decision_tree_accuracy(X_train, y_train, X_test, y_test), X_train, feature_dependence="independent")
shap_values = explainer.shap_values(X_train)


# In[1380]:


shap.summary_plot(shap_values, X_train)


# In[1381]:


explainer = shap.TreeExplainer(run_decision_tree_accuracy(X_train, y_train, X_test, y_test), X_train, feature_dependence="independent")
shap_values = explainer.shap_values(X_train)


# In[1382]:


shap.summary_plot(shap_values, X_train)


# In[1383]:


explainer = shap.LinearExplainer(run_regression_accuracy(X_train, y_train, X_test, y_test), X_train, feature_dependence="independent")
shap_values = explainer.shap_values(X_train)


# # Observations And Analysis

# By applying the algorithms where all the independent variables included doesn't produce any reasonable accuracy (accuracy scores between 47% and 63%) for any of the algorithms used.  Changing the size of the test and training dataset doesn't produce any significant changes in accuracy.  In light of this after exploring numerous possible solutions like changing the amount of training data and redefining the X and y variables by taking subsets of the dataset.  It was found that excluding variables X2 and X5 from the training dataset at a test size of 0.15 produced the best results with the corresponding decision tree at 74% accuracy and a random forest accuracy of 84%.  From varying the training data and testing data it was found that the accuracy score generally improved by increasing the amount of training data.  This indicates that when the training data is portioned below 85 to 80% the accuracy scores went up for random forests and decision trees.  If the size of the training data is below 80% it is probable that the algorithms are underfitting the data creating bias.

# ### Selecting The Best Algorithm

# Overall the best algorithm for most of the trials was a random forest where the test size is at 15%.  This indicates that the random forest algorithm has the highest amount of training data possible with a reasonable test size.  Varying the test size by a few % makes some significant difference as well since the decision boundries change enough to signficantly misclassify more datapoints.  15% seems to be the optimal value not only in having a reliable amount of datapoints but having decision boundries that best fits both the training and testing data.

# ### Explanation of the variables and their importance in determining the target variable

# The variables X2 and X5 were eliminated to give the best models.  This indicates that customers satisfaction depended very little on the courier and whether or not customers recieved the contents of their order as expected.  Customer cared most about how long it took for the order to be delivered (X1), the money spent on the order (X4) and the ease of using the ordering app.  According the the SHAP value analysis for the best models, the order of importance for the variables from most important to least important was X1>X6>X3>X4.  The time of the order was most important and the ease of which to make the order was the second most important followed by whether or not the customer ordered everything they wanted to order and the price they paid for the order.

# # Conclusion

# The best outcomes can best be influenced by focusing on the timeliness of the order, the ease of the ordering app (increase the app's ease of use), increasing the variety of products available at the cheapest price.
