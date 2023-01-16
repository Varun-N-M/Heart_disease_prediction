#Importing necessary libraries
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split,cross_val_score,KFold,GridSearchCV,learning_curve

from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import SelectFromModel

from sklearn.cluster import KMeans

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score


#Importing Dataset
df = pd.read_csv('framingham.csv')
df.rename(columns={'male':'Gender'}, inplace=True)
#Exploratory Data Analysis
print(df.head().to_string())
print( )
print(df.info())
print( )
print(df.describe())
print( )

print(df.isnull().sum())

#Data Cleaning
#Filling null values
#Categorizing all features into categorical and continuous
def grab_col_names(dataframe, car_th=10, cat_th=20):
    cat_col = [col for col in dataframe.columns if dataframe[col].dtype == 'O']
    num_but_cat = [col for col in dataframe.columns if
                   dataframe[col].nunique() < cat_th and dataframe[col].dtype != 'O']
    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].nunique() > cat_th and dataframe[col].dtype == 'O']

    cat_col = cat_col + num_but_cat
    cat_col = [col for col in cat_col if col not in cat_but_car]

    num_col = [col for col in dataframe.columns if dataframe[col].dtype != 'O']
    num_col = [col for col in num_col if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_col)}')
    print(f'num_cols: {len(num_col)}')

    return cat_col, num_col, cat_but_car
cat_col, num_col, cat_but_car = grab_col_names(df)

for i in num_col:
    df[i].fillna(df[i].mean(),inplace = True)

for i in cat_col:
    df[i].fillna(df[i].mode()[0],inplace = True)

print(df.isnull().sum())
corr = df.corr()
f, ax = plt.subplots(figsize = (10,10))
h_plot = sns.heatmap(corr,annot=True)

b_plot = df.boxplot()

def replace_outlier (mydf, col, method = 'Quartile',strategy = 'median' ):
    if method == 'Quartile':
        Q1 = mydf[col].quantile (0.25)
        Q2 =  mydf[col].quantile (0.50)
        Q3 = mydf[col].quantile (0.75)
        IQR =Q3 - Q1
        LW = Q1 - 1.5 * IQR
        UW = Q3 + 1.5 * IQR
    elif method == 'standard deviation':
        mean = mydf[col].mean()
        std = mydf[col].std()
        LW = mean - (2*std)
        UW = mean + (2*std)
    else:
        print('Pass a correct method')
#Printing all the outliers
    outliers = mydf.loc[(mydf[col]<LW) | (mydf[col]>UW),col]
    outliers_density = round(len(outliers)/ len(mydf),2)
    if len(outliers)==0:
        print(f'feature {col} does not have any outliers')
    else:
        print(f'feature {col} has outliers')
        print(f'Total number of outliers in this {col} is:'(len(outliers)))
        print(f'Outliers percentage in {col} is {outliers_density*100}%')
    if strategy=='median':
    #mydf.loc[ (mydf[col] < LW), col] = Q2 # used for first method
    #mydf.loc[ (mydf[col] > UW), col] = Q2 # used for first method
        mydf.loc[(mydf [col] < LW), col] = Q1 # second method.. the data may get currupted. so we are res
        mydf.loc[(mydf [col] > UW), col] = Q3 #second method.. as the outliers are more and not treated
    elif strategy == 'mean':
        mydf.loc[(mydf [col] < LW), col] = mean
        mydf.loc[(mydf [col] > UW), col] = mean
    else:
        print('Pass the correct strategy')
    return mydf

def odt_plots (mydf, col):
    f, (ax1, ax2) = plt.subplots (1,2,figsize=(25, 8))
    # descriptive statistic boxplot
    sns.boxplot (mydf [col], ax = ax1)
    ax1.set_title (col + ' boxplot')
    ax1.set_xlabel('values')
    ax1.set_ylabel('boxplot')
    #replacing the outliers
    mydf_out = replace_outlier (mydf, col)
    #plotting boxplot without outliers
    sns.boxplot (mydf_out[col], ax = ax2)
    ax2.set_title (col + 'boxplot')
    ax2.set_xlabel('values')
    ax2.set_ylabel('boxplot')
    # plt.show()
for col in df.drop('TenYearCHD',axis = 1).columns:
        odt_plots(df,col)

def PCA_1(x):
    n_comp = len(x.columns)
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    # Applying PCA
    for i in range(1, n_comp):
        pca = PCA(n_components=i)
        p_comp = pca.fit_transform(x)
        evr = np.cumsum(pca.explained_variance_ratio_)
        if evr[i - 1] > 0.9:
            n_components = i
            break
    print('Ecxplained varience ratio after pca is: ', evr)
    # creating a pcs dataframe
    col = []
    for j in range(1, n_components + 1):
        col.append('PC_' + str(j))
    pca_df = pd.DataFrame(p_comp, columns=col)
    return pca_df

transformed_df = PCA_1(df.drop('TenYearCHD',axis = 1))
transformed_df = transformed_df.join(df['TenYearCHD'],how = 'left')
print(transformed_df.head().to_string())

def train_and_test_split(data,t_col, testsize=0.3):
    x = data.drop(t_col, axis=1)
    y = data[t_col]
    return train_test_split(x,y,test_size=testsize, random_state=1)

def model_builder(model_name, estimator, data, t_col):
    x_train,x_test,y_train,y_test = train_and_test_split(data, t_col)
    estimator.fit(x_train, y_train)
    y_pred = estimator.predict(x_test)
    accuracy = accuracy_score(y_test,y_pred)
    return [model_name, accuracy]

def multiple_models(data,t_col):
    col_names = ['model_name' , 'accuracy_score' , ]
    result = pd.DataFrame(columns=col_names)
    result.loc[len(result)] = model_builder('LogisticRegression',LogisticRegression(),data,t_col)
    result.loc[len(result)] = model_builder('DecisionTreeClassifier',DecisionTreeClassifier(),data,t_col)
    result.loc[len(result)] = model_builder('KneighborClassifier',KNeighborsClassifier(),data,t_col)
    result.loc[len(result)] = model_builder('RandomForestClassifier',RandomForestClassifier(),data,t_col)
    result.loc[len(result)] = model_builder('SVC',SVC(),data,t_col)
    result.loc[len(result)] = model_builder('AdaBoostClassifier',AdaBoostClassifier(),data,t_col)
    result.loc[len(result)] = model_builder('GradientBoostingClassifier',GradientBoostingClassifier(),data,t_col)
    result.loc[len(result)] = model_builder('XGBClassifier',XGBClassifier(),data,t_col)
    return result.sort_values(by='accuracy_score',ascending=False)


def kfoldCV(x, y, fold=10):
    score_lr = cross_val_score(LogisticRegression(), x, y, cv=fold)
    score_dt = cross_val_score(DecisionTreeClassifier(), x, y, cv=fold)
    score_kn = cross_val_score(KNeighborsClassifier(), x, y, cv=fold)
    score_rf = cross_val_score(RandomForestClassifier(), x, y, cv=fold)
    score_svc = cross_val_score(SVC(), x, y, cv=fold)
    score_ab = cross_val_score(AdaBoostClassifier(), x, y, cv=fold)
    score_gb = cross_val_score(GradientBoostingClassifier(), x, y, cv=fold)
    score_xb = cross_val_score(XGBClassifier(), x, y, cv=fold)

    model_names = ['Logisticregression', 'DecisionTreeClassifier', 'KNeighborsClassifier', 'RandomForestClassifier',
                   'SVC', 'AdaBoostClassifier', 'GradientBoostingClassifier', 'XGBClassifier']
    scores = [score_lr, score_dt, score_kn, score_rf, score_svc, score_ab, score_gb, score_xb]
    result = []
    for i in range(len(model_names)):
        score_mean = np.mean(scores[i])
        score_std = np.std(scores[i])
        m_names = model_names[i]
        temp = [m_names, score_mean, score_std]
        result.append(temp)
    kfold_df = pd.DataFrame(result, columns=['model_names', 'cv_score', 'cv_std'])
    return kfold_df.sort_values(by='cv_score', ascending=False)


def tuning(x,y,fold = 10):
   #parameters grids for different models
    param_dtc = {'criterion':['gini', 'entropy', 'log_loss'],'max_depth':[3,5,7,9,11],'max_features':[1,2,3,4,5,6,7,'auto','log2', 'sqrt']}
    param_knn = {'weights':['uniform','distance'],'algorithm':['auto','ball_tree','kd_tree','brute']}
    param_svc = {'gamma':['scale','auto'],'C': [0.1,1,1.5,2]}
    param_rf = {'max_depth':[3,5,7,9,11],'max_features':[1,2,3,4,5,6,7,'auto','log2', 'sqrt'],'n_estimators':[50,100,150,200]}
    param_ab = {'n_estimators':[50,100,150,200],'learning_rate':[0.1,0.5,0.7,1,5,10,20,50,100]}
    param_gb = {'n_estimators':[50,100,150,200],'learning_rate':[0.1,0.5,0.7,1,5,10,20,50,100]}
    param_xb = {'eta':[0.1,0.5,10.7,1,5,10,20],'max_depth':[3,5,7,9,10],'gamma':[0,10,20,50],'reg_lambda':[0,1,3,5,7,10],'alpha':[0,1,3,5,7,10]}
    #Creating Model object
    tune_dtc = GridSearchCV(DecisionTreeClassifier(),param_dtc,cv=fold)
    tune_knn = GridSearchCV(KNeighborsClassifier(),param_knn,cv=fold)
    tune_svc = GridSearchCV(SVC(),param_svc,cv=fold)
    tune_rf = GridSearchCV(RandomForestClassifier(),param_rf,cv=fold)
    tune_ab = GridSearchCV(AdaBoostClassifier(),param_ab,cv=fold)
    tune_gb = GridSearchCV(GradientBoostingClassifier(),param_gb,cv=fold)
    tune_xb = GridSearchCV(XGBClassifier(),param_xb,cv=fold)
    #Model fitting
    tune_dtc.fit(x,y)
    tune_knn.fit(x,y)
    tune_svc.fit(x,y)
    tune_rf.fit(x,y)
    tune_ab.fit(x,y)
    tune_gb.fit(x,y)
    tune_xb.fit(x,y)

    tune = [tune_dtc,tune_knn,tune_svc,tune_rf,tune_ab,tune_gb,tune_xb]
    models = ['DTC','KNN','SVC','RF','AB','GB','XB']
    for i in range(len(tune)):
        print('model:',models[i])
        print('Best_params:',tune[i].best_params_)

tuning(df.drop('TenYearCHD',axis=1),df['TenYearCHD'])

def cv_post_hpt(x,y,fold = 10):
    score_lr = cross_val_score(LogisticRegression(),x,y,cv= fold)
    score_dt = cross_val_score(DecisionTreeClassifier(criterion= 'log_loss' ,max_depth= 3 ,max_features= 3),x,y,cv= fold)
    score_kn = cross_val_score(KNeighborsClassifier(weights ='uniform' ,algorithm ='auto' ),x,y,cv=fold)
    score_rf = cross_val_score(RandomForestClassifier(max_depth= 9, max_features=4 ,n_estimators= 100),x,y,cv=fold)
    score_svc = cross_val_score(SVC(gamma='scale' ,C= 0.1 ),x,y,cv=fold)
    score_ab = cross_val_score(AdaBoostClassifier(n_estimators= 200 ,learning_rate=0.1 ),x,y,cv=fold)
    score_gb = cross_val_score(GradientBoostingClassifier(n_estimators=50 ,learning_rate= 0.1),x,y,cv=fold)
    score_xb = cross_val_score(XGBClassifier(eta=0.5 ,max_depth= 10,gamma=0 ,reg_lambda = 10,alpha=10 ),x,y,cv=fold)

    model_names = ['LogisticRegression','DecisionTreeClassifier','KNeighborsClassifier','RandomForestClassifier','SVC','AdaBoostClassifier','GradientBoostingClassifier','XGBClassifier']
    scores = [score_lr,score_dt,score_kn,score_rf,score_svc,score_ab,score_gb,score_xb]
    result = []
    for i in range(len(model_names)):
        score_mean = np.mean(scores[i])
        score_std = np.std(scores[i])
        m_names = model_names[i]
        temp = [m_names,score_mean,score_std]
        result.append(temp)
    kfold_df = pd.DataFrame(result,columns=['model_names','cv_score','cv_std'])
    return kfold_df.sort_values(by='cv_score',ascending=False)

print(multiple_models(transformed_df,'TenYearCHD'))
print( )

print(kfoldCV(transformed_df.drop('TenYearCHD',axis=1),transformed_df['TenYearCHD']))
print( )

print(cv_post_hpt(transformed_df.drop('TenYearCHD',axis=1),transformed_df['TenYearCHD']))
print( )


labels = KMeans(n_clusters = 2,random_state=2)
clusters = labels.fit_predict(df.drop('TenYearCHD',axis=1))
sns.scatterplot(x=df['totChol'],y=df['TenYearCHD'],hue=clusters)

new_df = df.join(pd.DataFrame(clusters,columns=['cluster']),how = 'left')

new_f = new_df.groupby('cluster')['totChol'].agg(['mean','median'])

cluster_df = new_df.merge(new_f, on = 'cluster',how= 'left')
print(cluster_df.head().to_string())
print( )
print(multiple_models(cluster_df,'TenYearCHD'))
print( )
print(kfoldCV(cluster_df.drop('TenYearCHD',axis=1),cluster_df['TenYearCHD']))
print( )
print(cv_post_hpt(cluster_df.drop('TenYearCHD',axis=1),cluster_df['TenYearCHD']))

x = cluster_df.drop('TenYearCHD',axis=1)
y = cluster_df['TenYearCHD']

model = RandomForestClassifier()
model.fit(x, y)

selector = SelectFromModel(model, threshold='median')

selector.fit(x, y)

selected_indices = selector.get_support()

selected_features = x.iloc[:, selected_indices]

f_df = selected_features

c_to_drop = ['Gender','education','currentSmoker','BPMeds','prevalentStroke', 'prevalentHyp', 'diabetes']
n_df = cluster_df.drop(c_to_drop,axis=1)
sc = StandardScaler()
sc.fit_transform(n_df)

print(n_df.head().to_string())
print( )
print(multiple_models(n_df,'TenYearCHD'))
print( )
print(kfoldCV(n_df.drop('TenYearCHD',axis=1),n_df['TenYearCHD']))
print( )
print(cv_post_hpt(n_df.drop('TenYearCHD',axis=1),n_df['TenYearCHD']))

def generate_learning_curve(model_name,estimater,x,y):
    train_size,train_score,test_score = learning_curve(estimater,x,y,cv= 10)
#     print('train_size',train_size)
#     print('train_score',train_score)
#     print('test_score',test_score)
    train_score_mean = np.mean(train_score,axis=1)
    test_score_mean = np.mean(test_score,axis=1)
    plt.plot(train_size,train_score_mean, c = 'blue')
    plt.plot(train_size,test_score_mean, c = 'red')
    plt.xlabel('Samples')
    plt.ylabel('Scores')
    plt.title('Learning curve for '+model_name)
    plt.legend(('Training accuray','Testing accuracy'))

generate_learning_curve('LogisticRegression',LogisticRegression(),n_df.drop('TenYearCHD',axis=1),n_df['TenYearCHD'])
model_names = [LogisticRegression(), DecisionTreeClassifier(), KNeighborsClassifier(), RandomForestClassifier(), SVC(),
               AdaBoostClassifier(), GradientBoostingClassifier(), XGBClassifier()]
for i, model in enumerate(model_names):
#     print(i)
#     print(model_names[i])
    fig = plt.figure(figsize=(6,3))
    ax = fig.add_subplot(5,2,i+1)
    generate_learning_curve(type(model).__name__,model,n_df.drop('TenYearCHD',axis=1),n_df['TenYearCHD'])

plt.show()
