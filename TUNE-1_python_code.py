
# coding: utf-8

# In[1]:


#import required libraries
get_ipython().magic('matplotlib inline')
import matplotlib.pylab as plt
import seaborn as sns
from tsfresh import extract_features, extract_relevant_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report


# In[2]:


# We set the logger to Error level
# This is not recommend for normal use as you can oversee important Warning messages
import logging
logging.basicConfig(level=logging.ERROR)


# In[3]:


# read the file containing samples(60) of each instance of the time series signal
import pandas
#df = pandas.read_clipboard()
df = pandas.read_csv('data_240_percent_wise_noise.csv')


# In[4]:


extraction_settings = ComprehensiveFCParameters()
import pandas
X_act = extract_features(df, 
                     column_id='id', column_sort='time',
                     default_fc_parameters=extraction_settings,
                     impute_function= impute);


# In[5]:


X = X_act.copy()
X.head()

min(X['F_x__abs_energy'])fea_list = list(X)

for f in fea_list:
    X[f] = (X[f] - min(X[f])) / (max(X[f]) - min(X[f]))
# In[6]:


#feature normalization - http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.normalize.html
from sklearn import preprocessing
X = preprocessing.normalize(X, axis=0, norm='l2')
X = pandas.DataFrame(X)
X.head()


# In[7]:


X = X.dropna(axis=1, how='any')


# In[8]:


import numpy as np
y_act = pandas.read_csv('label_data_240_percent_wise_noise.csv')

y_act = np.array(y_act['label'])
y_act = pandas.Series(y_act, index = np.arange(0,240))

y = y_act.copy()


# In[9]:


X['label'] = y

train = X[0:180].copy()
test = X[180:240].copy()


#train = X[60:240].copy()
#train = train.append(X[120:240].copy())
#test = X[0:60].copy()


# In[10]:


y_train = train['label']
x_train = train.drop('label', 1)

y_test = test['label']
x_test = test.drop('label', 1)


# In[11]:


# Removing features with low variance
from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold()
sel.fit_transform(x_train)

ind_varThresh = sel.get_support(indices=True)

len(sel.get_support(indices=True))

X_filtered = x_train.iloc[:,ind_varThresh]
X_filtered.head()


# In[12]:


import numpy as np
x_a = np.array(X_filtered).copy()
y_a = np.array(y_train).copy()


# In[13]:


type(y_train)


# In[14]:


# Recursive Feature Elimination
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
rfe = RFE(model, 1)
rfe = rfe.fit(X_filtered, y_train)

idx = rfe.ranking_
rank_rfe = {}
p = 0;
for q in range(X_filtered.columns.size):
    if X_filtered.columns[q] not in rank_rfe:
        rank_rfe[X_filtered.columns[q]] = idx[q]
        
#rank_rfe

sort_ind = sorted(rank_rfe, key=rank_rfe.get, reverse = False)


nf = []
acc_all_svm = []
acc_all_knn = []
acc_all_lo_reg = []
acc_all_nb = []

for it in range(int(X_filtered.columns.size/10)):
    num_fea = 10 * (it+1)
    nf.append(num_fea)
    col = sort_ind[0:num_fea]
    
    selected_features_train = X_filtered.ix[:, col].copy()
    selected_features_test = x_test.ix[:, col].copy()

    #SVM classifier (linear)
    clf = svm.LinearSVC()
    clf.fit(selected_features_train, y_train)
    y_predict_svm = clf.predict(selected_features_test)
    acc_svm = accuracy_score(y_test, y_predict_svm)
    acc_all_svm.append(acc_svm)
    
    
    #SVM classifier (linear)
    from sklearn import svm
    clf = GaussianNB()
    clf.fit(selected_features_train, y_train)
    y_predict_nb = clf.predict(selected_features_test)
    acc_nb = accuracy_score(y_test, y_predict_nb)
    acc_all_nb.append(acc_nb)

   

    #kNN classifier
    clf_knn = KNeighborsClassifier(n_neighbors=7)
    clf_knn.fit(selected_features_train, y_train)
    y_predict_knn = clf_knn.predict(selected_features_test)
    acc_knn = accuracy_score(y_test, y_predict_knn)
    acc_all_knn.append(acc_knn)

    
    
    model = LogisticRegression()
    model.fit(selected_features_train, y_train)
    y_predict_lo_reg= model.predict(selected_features_test)
    acc_lo_reg = accuracy_score(y_test, y_predict_lo_reg)
    acc_all_lo_reg.append(acc_lo_reg)
    
    

plot1, = plt.plot(nf,acc_all_knn, 'r')
#plot2, = plt.plot(nf,acc_all_svm,  'r')
plot3, = plt.plot(nf,acc_all_lo_reg,  'g')
plot4, = plt.plot(nf,acc_all_nb,  'b')

# make legend
plt.legend((plot1, plot3, plot4), ('kNN', 'LogisticRegression', 'Naive Bayes'), loc='upper right', shadow=True)
plt.xlabel('Number of features')
plt.ylabel('Accuracy')
plt.title('Feature selection using RFE')
plt.show()


# In[15]:


#for average ranking purpose(not used here)
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
rank_SelectKBest_f_classif = {}
for p in range(X_filtered.columns.size):
        X_new = SelectKBest(f_classif, k=p+1).fit(X_filtered, y_train)
        new = X_new.get_support(indices=True)
        for q in new:
            if X_filtered.columns[q] not in rank_SelectKBest_f_classif:
                rank_SelectKBest_f_classif[X_filtered.columns[q]] = p+1
                
#rank_SelectKBest_f_classif
sort_ind = sorted(rank_SelectKBest_f_classif, key=rank_SelectKBest_f_classif.get, reverse = False)


nf = []
acc_all_svm = []
acc_all_knn = []
acc_all_lo_reg = []
for it in range(int(X_filtered.columns.size/10)):
    num_fea = 10 * (it+1)
    nf.append(num_fea)
    col = sort_ind[0:num_fea]
    
    selected_features_train = X_filtered.ix[:, col].copy()
    selected_features_test = x_test.ix[:, col].copy()

    #SVM classifier (linear)
    from sklearn import svm
    clf = GaussianNB()
    clf.fit(selected_features_train, y_train)

    y_predict_svm = clf.predict(selected_features_test)

    from sklearn.metrics import accuracy_score
    acc_svm = accuracy_score(y_test, y_predict_svm)
    acc_all_svm.append(acc_svm)

    from sklearn import svm
    from sklearn.neural_network import MLPClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier


    clf_knn = KNeighborsClassifier(n_neighbors=7)
    clf_knn.fit(selected_features_train, y_train)
    y_predict_knn = clf_knn.predict(selected_features_test)
    acc_knn = accuracy_score(y_test, y_predict_knn)
    acc_all_knn.append(acc_knn)

    
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    # Train the model using the training sets and check score
    model.fit(selected_features_train, y_train)
    model.score(selected_features_train, y_train)
    y_predict_lo_reg= model.predict(selected_features_test)
    acc_lo_reg = accuracy_score(y_test, y_predict_lo_reg)
    acc_all_lo_reg.append(acc_lo_reg)
    
    
    
# f_classif
plot1, = plt.plot(nf,acc_all_knn, 'r')
plot2, = plt.plot(nf,acc_all_svm,  'b')
plot3, = plt.plot(nf,acc_all_lo_reg,  'g')

# make legend
plt.legend((plot1, plot2, plot3), ('kNN', 'Naive Bayes', 'LogisticRegression'), loc='upper right', shadow=True)
plt.xlabel('Number of features')
plt.ylabel('Accuracy')
plt.title('Feature selection using SelectKBest')
plt.show()


# In[16]:


from skfeature.function.statistical_based import gini_index
score = gini_index.gini_index(x_a, y_a)
idx = gini_index.feature_ranking(score)

rank_gini_index = {}
p = 0;
for q in idx:
    if X_filtered.columns[q] not in rank_gini_index:
        rank_gini_index[X_filtered.columns[q]] = p+1
        p=p+1;
        
#rank_gini_index

sort_ind = sorted(rank_gini_index, key=rank_gini_index.get, reverse = False)


nf = []
acc_all_svm = []
acc_all_knn = []
acc_all_lo_reg = []
for it in range(int(X_filtered.columns.size/10)):
    num_fea = 10 * (it+1)
    nf.append(num_fea)
    col = sort_ind[0:num_fea]
    
    selected_features_train = X_filtered.ix[:, col].copy()
    selected_features_test = x_test.ix[:, col].copy()

    #SVM classifier (linear)
    from sklearn import svm
    clf = GaussianNB()
    clf.fit(selected_features_train, y_train)

    y_predict_svm = clf.predict(selected_features_test)

    from sklearn.metrics import accuracy_score
    acc_svm = accuracy_score(y_test, y_predict_svm)
    acc_all_svm.append(acc_svm)

    from sklearn import svm
    from sklearn.neural_network import MLPClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier


    clf_knn = KNeighborsClassifier(n_neighbors=7)
    clf_knn.fit(selected_features_train, y_train)
    y_predict_knn = clf_knn.predict(selected_features_test)
    acc_knn = accuracy_score(y_test, y_predict_knn)
    acc_all_knn.append(acc_knn)

    
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    # Train the model using the training sets and check score
    model.fit(selected_features_train, y_train)
    model.score(selected_features_train, y_train)
    y_predict_lo_reg= model.predict(selected_features_test)
    acc_lo_reg = accuracy_score(y_test, y_predict_lo_reg)
    acc_all_lo_reg.append(acc_lo_reg)
    
    
    
# f_classif
plot1, = plt.plot(nf,acc_all_knn,  'r')
plot2, = plt.plot(nf,acc_all_svm,  'b')
plot3, = plt.plot(nf,acc_all_lo_reg,  'g')

# make legend
plt.legend((plot1, plot2, plot3), ('kNN', 'Naive Bayes', 'LogisticRegression'), loc='upper right', shadow=True)
plt.xlabel('Number of features')
plt.ylabel('Accuracy')
plt.title('Feature selection using gini_index')
plt.show()


# In[17]:


# average ranking of all

from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

avg = {}
for k,v in rank_SelectKBest_f_classif.items():
    avg[k] =(rank_SelectKBest_f_classif[k]+rank_gini_index[k]+rank_rfe[k])/3
#avg

sort_ind = sorted(avg, key=avg.get, reverse = False)


nf = []
acc_all_svm = []
acc_all_knn = []
acc_all_lo_reg = []
for it in range(int(X_filtered.columns.size/10)):
    num_fea = 10 * (it+1)
    nf.append(num_fea)
    col = sort_ind[0:num_fea]
    
    selected_features_train = X_filtered.ix[:, col].copy()
    selected_features_test = x_test.ix[:, col].copy()

    #SVM classifier (linear)
    
    clf = GaussianNB()
    clf.fit(selected_features_train, y_train)

    y_predict_svm = clf.predict(selected_features_test)

    
    acc_svm = accuracy_score(y_test, y_predict_svm)
    acc_all_svm.append(acc_svm)
    
    #print(classification_report(y_test, y_predict_svm))
    #print(confusion_matrix(y_test, y_predict_svm))
    #print(y_predict_svm)

    

    clf_knn = KNeighborsClassifier(n_neighbors=7)
    clf_knn.fit(selected_features_train, y_train)
    y_predict_knn = clf_knn.predict(selected_features_test)
    acc_knn = accuracy_score(y_test, y_predict_knn)
    acc_all_knn.append(acc_knn)

    #print(classification_report(y_test, y_predict_knn))
    #print(confusion_matrix(y_test, y_predict_knn))
    #print(y_predict_knn)
    
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    # Train the model using the training sets and check score
    model.fit(selected_features_train, y_train)
    model.score(selected_features_train, y_train)
    y_predict_lo_reg= model.predict(selected_features_test)
    acc_lo_reg = accuracy_score(y_test, y_predict_lo_reg)
    acc_all_lo_reg.append(acc_lo_reg)
    
    #print(classification_report(y_test, y_predict_lo_reg))
    #print(confusion_matrix(y_test, y_predict_lo_reg))
    #print(y_predict_lo_reg)
    
    
# f_classif
plot1, = plt.plot(nf,acc_all_knn,  'r')
plot2, = plt.plot(nf,acc_all_svm,  'b')
plot3, = plt.plot(nf,acc_all_lo_reg,  'g')

# make legend
plt.legend((plot1, plot2, plot3), ('kNN', 'Naive Bayes', 'LogisticRegression'), loc='upper right', shadow=True)
plt.xlabel('Number of features')
plt.ylabel('Accuracy')
plt.title('Average rank of all 3')
plt.show()


# In[18]:


max(acc_all_lo_reg)


# In[19]:


len(sort_ind)


# In[20]:


col = sort_ind[0:20]
selected_features_train = X_filtered.ix[:, col].copy()
selected_features_test = x_test.ix[:, col].copy()


# In[21]:


selected_features_test['label'] = y_test


# In[22]:


n_0 = np.concatenate((np.arange(0,60,12), np.arange(1,60,12)))
n_10 = np.concatenate((np.arange(2,60,12), np.arange(3,60,12)))
n_20 = np.concatenate((np.arange(4,60,12), np.arange(5,60,12)))
n_30 = np.concatenate((np.arange(6,60,12), np.arange(7,60,12)))
n_40 = np.concatenate((np.arange(8,60,12), np.arange(9,60,12)))
n_50 = np.concatenate((np.arange(10,60,12), np.arange(11,60,12)))


# In[23]:


n_0_test = selected_features_test.iloc[n_0, :].copy()
n_10_test = selected_features_test.iloc[n_10, :].copy()
n_20_test = selected_features_test.iloc[n_20, :].copy()
n_30_test = selected_features_test.iloc[n_30, :].copy()
n_40_test = selected_features_test.iloc[n_40, :].copy()
n_50_test = selected_features_test.iloc[n_50, :].copy()
selected_features_test = selected_features_test.drop('label', 1)


# In[24]:


n_acc_knn = []
from sklearn.neighbors import KNeighborsClassifier
clf_knn = KNeighborsClassifier(n_neighbors=7)
clf_knn.fit(selected_features_train, y_train)

acc_knn = accuracy_score(n_0_test['label'], clf_knn.predict(n_0_test.drop('label', 1)))
n_acc_knn.append(acc_knn)

acc_knn = accuracy_score(n_10_test['label'], clf_knn.predict(n_10_test.drop('label', 1)))
n_acc_knn.append(acc_knn)

acc_knn = accuracy_score(n_20_test['label'], clf_knn.predict(n_20_test.drop('label', 1)))
n_acc_knn.append(acc_knn)

acc_knn = accuracy_score(n_30_test['label'], clf_knn.predict(n_30_test.drop('label', 1)))
n_acc_knn.append(acc_knn)

acc_knn = accuracy_score(n_40_test['label'], clf_knn.predict(n_40_test.drop('label', 1)))
n_acc_knn.append(acc_knn)

acc_knn = accuracy_score(n_50_test['label'], clf_knn.predict(n_50_test.drop('label', 1)))
n_acc_knn.append(acc_knn)


# In[25]:


n_acc_nb = []
clf = GaussianNB()
clf.fit(selected_features_train, y_train)


acc_nb = accuracy_score(n_0_test['label'], clf.predict(n_0_test.drop('label', 1)))
n_acc_nb.append(acc_nb)

acc_nb = accuracy_score(n_10_test['label'], clf.predict(n_10_test.drop('label', 1)))
n_acc_nb.append(acc_nb)

acc_nb = accuracy_score(n_20_test['label'], clf.predict(n_20_test.drop('label', 1)))
n_acc_nb.append(acc_nb)

acc_nb = accuracy_score(n_30_test['label'], clf.predict(n_30_test.drop('label', 1)))
n_acc_nb.append(acc_nb)

acc_nb = accuracy_score(n_40_test['label'], clf.predict(n_40_test.drop('label', 1)))
n_acc_nb.append(acc_nb)

acc_nb = accuracy_score(n_50_test['label'], clf.predict(n_50_test.drop('label', 1)))
n_acc_nb.append(acc_nb)


# In[26]:


n_acc_lg = []
from sklearn.linear_model import LogisticRegression
clf_lg = LogisticRegression()
clf_lg.fit(selected_features_train, y_train)


acc_lg = accuracy_score(n_0_test['label'], clf_lg.predict(n_0_test.drop('label', 1)))
n_acc_lg.append(acc_lg)

acc_lg = accuracy_score(n_10_test['label'], clf_lg.predict(n_10_test.drop('label', 1)))
n_acc_lg.append(acc_lg)

acc_lg = accuracy_score(n_20_test['label'], clf_lg.predict(n_20_test.drop('label', 1)))
n_acc_lg.append(acc_lg)

acc_lg = accuracy_score(n_30_test['label'], clf_lg.predict(n_30_test.drop('label', 1)))
n_acc_lg.append(acc_lg)

acc_lg = accuracy_score(n_40_test['label'], clf_lg.predict(n_40_test.drop('label', 1)))
n_acc_lg.append(acc_lg)

acc_lg = accuracy_score(n_50_test['label'], clf_lg.predict(n_50_test.drop('label', 1)))
n_acc_lg.append(acc_lg)


# In[27]:


clf.predict(n_20_test.drop('label', 1))


# In[28]:


noi = [0, 10, 20, 30, 40, 50]


# In[29]:


plot1, = plt.plot(noi,n_acc_knn, marker='o')
plot2, = plt.plot(noi,n_acc_nb,  'r', marker='o')
plot3, = plt.plot(noi,n_acc_lg,  'g', marker='o')

# make legend
plt.legend((plot1, plot2, plot3), ('kNN', 'Naive Bayes', 'Logistic Regression'), loc='upper right', shadow=True)
plt.xlabel('% of artificial noise')
plt.ylabel('Accuracy')
plt.title('Noise vs Accuracy')
plt.show()

# Four axes, returned as a 2-d array
f, axarr = plt.subplots(2, 2)
axarr[0, 0].plot(x, y)
axarr[0, 0].set_title('Axis [0,0]')
axarr[0, 1].scatter(x, y)
axarr[0, 1].set_title('Axis [0,1]')
axarr[1, 0].plot(x, y ** 2)
axarr[1, 0].set_title('Axis [1,0]')
axarr[1, 1].scatter(x, y ** 2)
axarr[1, 1].set_title('Axis [1,1]')
# Fine-tune figure; hide x ticks for top plots and y ticks for right plots
plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)
# In[30]:


col = sort_ind[0:20]
selected_features_X = X.ix[:, col].copy()


# In[31]:


#GaussianNB()KNeighborsClassifier(n_neighbors=7)LogisticRegression()svm.LinearSVC()
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
acc_knn_pt = []
k_knn = []
for i in range(80):
    kf = KFold(n_splits=4)
    avg_acc_svm = []
    for train_ind, test_ind in kf.split(selected_features_X):
        clf_svm = KNeighborsClassifier(n_neighbors=i+1)
        clf_svm.fit(selected_features_X.iloc[train_ind], y[train_ind])
        y_predict_svm = clf_svm.predict(selected_features_X.iloc[test_ind])
        acc_svm = accuracy_score(y[test_ind], y_predict_svm)
        #print(acc_svm)
        avg_acc_svm.append(acc_svm)

    avg_acc_svm = sum(avg_acc_svm)/4
    acc_knn_pt.append(avg_acc_svm)
    k_knn.append(i+1)
    
plot1, = plt.plot(k_knn,acc_knn_pt, marker='o')

# make legend
#plt.legend((plot1, plot2, plot3), ('kNN', 'Naive Bayes', 'Logistic Regression'), loc='upper right', shadow=True)
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.title('kNN')
plt.show()


# In[32]:


from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }
clf = GridSearchCV(LogisticRegression(penalty='l2'), param_grid)


# In[33]:


C_param_range = [0.001,0.01,0.1,1,10,100]

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
acc_lg_pt = []
for i in C_param_range:
    kf = KFold(n_splits=4)
    avg_acc_svm = []
    for train_ind, test_ind in kf.split(selected_features_X):
        clf_svm = LogisticRegression(penalty = 'l2', C = i,random_state = 0)
        clf_svm.fit(selected_features_X.iloc[train_ind], y[train_ind])
        y_predict_svm = clf_svm.predict(selected_features_X.iloc[test_ind])
        acc_svm = accuracy_score(y[test_ind], y_predict_svm)
        #print(acc_svm)
        avg_acc_svm.append(acc_svm)

    avg_acc_svm = sum(avg_acc_svm)/4
    acc_lg_pt.append(avg_acc_svm)
    #k_knn.append(i+1)
    
plot1, = plt.plot(C_param_range,acc_lg_pt, 'g',marker='o')

# make legend
#plt.legend((plot1, plot2, plot3), ('kNN', 'Naive Bayes', 'Logistic Regression'), loc='upper right', shadow=True)
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.title('Logistic regression parameter tuning')
plt.show()





# In[34]:


#SVM classifier (linear)
from sklearn.metrics import precision_recall_fscore_support
from sklearn import svm
clf = svm.LinearSVC()
clf.fit(selected_features_train, y_train)

y_predict_svm = clf.predict(selected_features_test)

from sklearn.metrics import accuracy_score
acc_svm = accuracy_score(y_test, y_predict_svm)
print(acc_svm)
    
from sklearn.metrics import confusion_matrix
print(classification_report(y_test, clf.predict(selected_features_test)))
print(confusion_matrix(y_test, clf.predict(selected_features_test)))
print(clf.predict(selected_features_test))
ap = precision_recall_fscore_support(y_test, clf.predict(selected_features_test),average='weighted')


# In[35]:


#GaussianNB()KNeighborsClassifier(n_neighbors=7)LogisticRegression()svm.LinearSVC()
from sklearn.metrics import precision_recall_fscore_support
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
kf = KFold(n_splits=4)
avg_acc_svm = []
avg_p_svm = []
avg_r_svm = []
avg_f_svm = []
for train_ind, test_ind in kf.split(selected_features_X):
    #clf_svm = LogisticRegression(penalty = 'l2', C = 0.01,random_state = 0)
    
    clf_svm = GaussianNB()
    
    clf_svm.fit(selected_features_X.iloc[train_ind], y[train_ind])
    y_predict_svm = clf_svm.predict(selected_features_X.iloc[test_ind])
    acc_svm = accuracy_score(y[test_ind], y_predict_svm)
    avg_acc_svm.append(acc_svm)
    prf = precision_recall_fscore_support(y[test_ind], y_predict_svm,average=None)
    avg_p_svm.append(prf[0])
    avg_r_svm.append(prf[1])
    avg_f_svm.append(prf[2])
avg_acc_svm = sum(avg_acc_svm)/4
avg_p_svm = sum(avg_p_svm)/4
avg_r_svm = sum(avg_r_svm)/4
avg_f_svm = sum(avg_f_svm)/4

print('acc' , avg_acc_svm)
print('p' , avg_p_svm)
print('r' , avg_r_svm)
print('f1' , avg_f_svm)


# In[36]:


#GaussianNB()KNeighborsClassifier(n_neighbors=7)LogisticRegression()svm.LinearSVC()
from sklearn.metrics import precision_recall_fscore_support
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
kf = KFold(n_splits=4)
avg_acc_svm = []
avg_p_svm = []
avg_r_svm = []
avg_f_svm = []
for train_ind, test_ind in kf.split(selected_features_X):
    clf_svm = LogisticRegression(penalty = 'l2', C = 0.01,random_state = 0)    
    clf_svm.fit(selected_features_X.iloc[train_ind], y[train_ind])
    y_predict_svm = clf_svm.predict(selected_features_X.iloc[test_ind])
    acc_svm = accuracy_score(y[test_ind], y_predict_svm)
    avg_acc_svm.append(acc_svm)
    prf = precision_recall_fscore_support(y[test_ind], y_predict_svm,average=None)
    avg_p_svm.append(prf[0])
    avg_r_svm.append(prf[1])
    avg_f_svm.append(prf[2])
avg_acc_svm = sum(avg_acc_svm)/4
avg_p_svm = sum(avg_p_svm)/4
avg_r_svm = sum(avg_r_svm)/4
avg_f_svm = sum(avg_f_svm)/4

print('acc' , avg_acc_svm)
print('p' , avg_p_svm)
print('r' , avg_r_svm)
print('f1' , avg_f_svm)


# In[37]:


#GaussianNB()KNeighborsClassifier(n_neighbors=7)LogisticRegression()svm.LinearSVC()
from sklearn.metrics import precision_recall_fscore_support
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
kf = KFold(n_splits=4)
avg_acc_svm = []
avg_p_svm = []
avg_r_svm = []
avg_f_svm = []
for train_ind, test_ind in kf.split(selected_features_X):
    clf_svm = KNeighborsClassifier(n_neighbors=1)
    clf_svm.fit(selected_features_X.iloc[train_ind], y[train_ind])
    y_predict_svm = clf_svm.predict(selected_features_X.iloc[test_ind])
    acc_svm = accuracy_score(y[test_ind], y_predict_svm)
    avg_acc_svm.append(acc_svm)
    prf = precision_recall_fscore_support(y[test_ind], y_predict_svm,average=None)
    avg_p_svm.append(prf[0])
    avg_r_svm.append(prf[1])
    avg_f_svm.append(prf[2])
avg_acc_svm = sum(avg_acc_svm)/4
avg_p_svm = sum(avg_p_svm)/4
avg_r_svm = sum(avg_r_svm)/4
avg_f_svm = sum(avg_f_svm)/4

print('acc' , avg_acc_svm)
print('p' , avg_p_svm)
print('r' , avg_r_svm)
print('f1' , avg_f_svm)

