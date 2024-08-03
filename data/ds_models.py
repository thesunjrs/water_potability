#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas            as pd
import numpy             as np
import matplotlib.pyplot as plt
import seaborn           as sns
import pydotplus

from six             import StringIO
from IPython.display import Image

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline                import Pipeline
from sklearn.impute                  import SimpleImputer, KNNImputer
from sklearn.preprocessing           import StandardScaler, OneHotEncoder, normalize
from sklearn.model_selection         import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics                 import mean_squared_error, mean_squared_log_error, accuracy_score
from sklearn.metrics                 import plot_confusion_matrix, classification_report

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors    import KNeighborsClassifier
from sklearn.naive_bayes  import GaussianNB
from sklearn.tree         import DecisionTreeRegressor, DecisionTreeClassifier, export_graphviz
from sklearn.ensemble     import BaggingClassifier, RandomForestClassifier
from sklearn.ensemble     import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm          import SVC

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


def ds_models(model, X, y, model_name, output, imputer=False, scale=False, clean=False, fi=False, tree=False, cvs=False, vc=False):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=0.25, random_state=1)

    if imputer == True:
        
        knn     = KNNImputer()
        X_train = knn.fit_transform(X_train)
        X_test  = knn.transform(X_test)
    
    if scale == True:
        
        scaler  = StandardScaler()  
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)  
        X_test  = scaler.transform(X_test) 
        
    if clean == True:
    
        cont_features = [col for col in X.columns if X[col].dtype in [np.float64, np.int64]]        
        X_train_cont  = X_train.loc[:, cont_features]
        X_test_cont   = X_test.loc[:, cont_features]
        
        impute          = SimpleImputer(strategy='median') 
        X_train_imputed = impute.fit_transform(X_train_cont)
        X_test_imputed  = impute.transform(X_test_cont)
        
        features_cat = [col for col in X.columns if X[col].dtype in [np.object]]
        X_train_cat  = X_train.loc[:, features_cat]
        X_test_cat   = X_test.loc[:, features_cat]
        
        X_train_cat.fillna(value='missing', inplace=True)
        X_test_cat.fillna(value='missing', inplace=True)
        
        scaler                 = StandardScaler()
        X_train_imputed_scaled = scaler.fit_transform(X_train_imputed)
        X_test_imputed_scaled  = scaler.transform(X_test_imputed)
        
        ohe         = OneHotEncoder(handle_unknown='ignore')
        X_train_ohe = ohe.fit_transform(X_train_cat)
        X_test_ohe  = ohe.transform(X_test_cat)

        columns      = ohe.get_feature_names(input_features=X_train_cat.columns)
        cat_train_df = pd.DataFrame(X_train_ohe.todense(), columns=columns)
        cat_test_df  = pd.DataFrame(X_test_ohe.todense(), columns=columns)
                                    
        X_train     = pd.concat([pd.DataFrame(X_train_imputed_scaled), cat_train_df], axis=1)
        X_test_all  = pd.concat([pd.DataFrame(X_test_imputed_scaled), cat_test_df], axis=1)
    
    model.fit(X_train, y_train)
        
    if output == 'class':
                
        print('\033[1m' + model_name + ' Train Data Confusion Matrix:\n')
        plot_confusion_matrix(model, X_train, y_train, cmap=plt.cm.Blues)
        plt.show()
        print('\033[1m' + model_name + ' Test Data Confusion Matrix:\n')
        plot_confusion_matrix(model, X_test, y_test, cmap=plt.cm.Blues)
        plt.show()        
        print('\033[1m' + model_name + ' Train Report:\n' + '\033[0m')
        print(classification_report(y_train, model.predict(X_train)))
        print('\033[1m' + model_name + ' Test Report:\n' + '\033[0m')
        print(classification_report(y_test, model.predict(X_test)))
    
    if output == 'reg':
                        
        print('\033[1m' + model_name + ' Training r^2:\n' + '\033[0m', 
        model.score(X_train, y_train))
        print('\n' + '\033[1m' + model_name + ' Test r^2:\n' + '\033[0m', 
        model.score(X_test_all, y_test))
        print('\n' + '\033[1m' + model_name + ' Training MSE:\n' + '\033[0m', 
        mean_squared_error(y_train, model.predict(X_train)))
        print('\n' + '\033[1m' + model_name + ' Test MSE:\n' + '\033[0m', 
        mean_squared_error(y_test, model.predict(X_test_all)))
        print('\n' + '\033[1m' + model_name + ' Training RMSE:\n' + '\033[0m', 
        mean_squared_error(y_train, model.predict(X_train))**0.5)
        print('\n' + '\033[1m' + model_name + ' Test RMSE:\n' + '\033[0m', 
        mean_squared_error(y_test, model.predict(X_test_all))**0.5)
        
    if fi == True:
        
        print('\033[1m' + 'Feature Importances:' + '\n' + '\033[0m')
        print(model.feature_importances_)
        plt.barh(X.columns, model.feature_importances_)
        plt.title('Feature Importances')
        
    if tree == True:         

        dot_data = StringIO()
        export_graphviz(model, out_file=dot_data, rounded=True, special_characters=True)
        graph    = pydotplus.graph_from_dot_data(dot_data.getvalue())
        print('\n' + '\033[1m' + 'Decision Tree:' + '\033[0m')
        display(Image(graph.create_png()))
        
    if cvs == True:

        depth_range = range(1,10)
        val = []
        for depth in depth_range:
            mod = DecisionTreeClassifier(max_depth = depth)
            depth_score = cross_val_score(mod, X, y, cv = 10)
            val.append(depth_score.mean())
        print('\n' + '\033[1m' + 'Cross Validation Scores:' + '\n' + '\033[0m')
        print(val)
        print('\n' + '\033[1m' + 'Cross Validation Curve:' + '\n' + '\033[0m')
        plt.figure(figsize = (10,10))
        plt.plot(depth_range, val)
        plt.xlabel('Range Of Depth')
        plt.ylabel('Cross Validated Values')
        plt.show()
        
    if vc == True:
        
        depth_range = range(1,10)
        mse = []
        for depth in depth_range:
            mod = DecisionTreeRegressor(max_depth = depth)
            depth_score = cross_val_score(mod, X, y, scoring = 'neg_mean_squared_error', cv = 6)
            mse.append(depth_score.mean())
        print('\n' + '\033[1m' + 'MSE:' + '\n' + '\033[0m')
        print(mse)
        print('\n' + '\033[1m' + 'Validation Curve:' + '\n' + '\033[0m')
        mse = [abs(number) for number in mse]
        plt.figure(figsize = (10,10))
        plt.plot(depth_range, mse)
        plt.xlabel('Range Of Depth')
        plt.ylabel('MSE')
        plt.show()
        
    return model

