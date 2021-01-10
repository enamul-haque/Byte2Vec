#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 14:48:43 2017
@purpose: classification
@author: Md Enamul Haque
@inistitute: University of Louisian at Lafayette
"""
from sklearn.model_selection import cross_validate
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, hamming_loss
from sklearn.metrics import precision_score, recall_score, f1_score
import glob
from sklearn.model_selection import cross_val_score
from sklearn import svm
import warnings
from sklearn import preprocessing
import itertools
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from matplotlib import rcParams
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
import sys
#from yellowbrick.classifier import ConfusionMatrix
warnings.filterwarnings("ignore")


# generate a confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plot_conf_matrix_and_save(cnf_matrix,class_names,size,sample_size,save_at):
    
    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion matrix, without normalization')
    
    plt.savefig(save_at+'general_conf_for_vec_size_'+str(size)+'_sample_size_'+str(sample_size)+'.eps')
    
    # Plot normalized confusion matrix
    plt.figure()
    
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')
    
    
    plt.savefig(save_at+'normalized_conf_for_vec_size_'+str(size)+'_sample_size_'+str(sample_size)+'.eps')
    plt.show()
        
def main(s,ratio,k,cross,conf_mat_plot):
    
    save_at = './results_data/'
    sizes = list(np.arange(5,105,5))
    sample_size = s
    f = open(save_at+'results_sample_size_'+str(sample_size)+'.txt','w')
    f.write('vec_length'+ '\t' + 'accuracy' + '\t' + 'precision' + '\t'+ 'recall' + '\t'+'hamming_loss' + '\n')
    path = './feature_data/'
    for i in range(len(sizes)):
        
        last_data_column = sizes[i] #4095
        class_column = sizes[i] #4096 
        data = pd.read_csv(path+'feature_data_vec_' + str(class_column) +'_sample_'+str(sample_size)+'.csv', low_memory=False)
        #orig_data = data
        # select all pdf type rows: data.loc[data.data_type=='pdf']
        type_count = data.groupby('data_type').size()
        #print(type_count)
        valid_types = type_count.loc[type_count >= sample_size]
        data = data.loc[data.data_type.isin(valid_types.index)]
        #select first "sample_size" rows of type 0
        #data = orig_data.loc[orig_data.data_type==valid_types.index[0]].head(sample_size)
        #for k in range(1,len(valid_types)):
        #    data=data.append(orig_data.loc[orig_data.data_type==valid_types.index[k]].head(sample_size))
        # randomize the data
        data = data.iloc[np.random.permutation(len(data))]
        # reset the index
        data = data.reset_index(drop=True)
        
        #data = data.drop('data_type', 1) # remove data type
        data = data.fillna(method='ffill')
        sz = data.shape    
        train = data.iloc[:int(sz[0] * ratio), :]
        test = data.iloc[int(sz[0] * ratio):, :]
        X = train.iloc[:,0:last_data_column]
        y = train.data_type
        # separate feature and label 
        test_X = test.iloc[:,0:last_data_column]
        # label column
        test_Y = test.data_type
        model = KNeighborsClassifier(n_neighbors=k).fit(X,y)
        y_hat = model.predict(test_X)
        # 10 fold cross validation
        accuracy = cross_val_score(model, X, y, cv=cross, scoring='accuracy').mean()
        scoring = ['precision_macro', 'recall_macro']
        clf = KNeighborsClassifier(n_neighbors=k) 
        scores = cross_validate(clf, X, y, scoring=scoring, cv=cross, return_train_score=True)
        precision = scores['test_precision_macro'].mean()
        recall = scores['test_recall_macro'].mean()
        print("*********Results**********")
        print("Avg. Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        f.write(str(sizes[i])+ '\t' + str(accuracy) + '\t' + str(precision) + '\t' + str(recall) + '\t'+ str(hamming_loss(test_Y, y_hat)) + '\n')
        print("Done for vector length:"+ str(sizes[i]) + " and sample size:"+str(sample_size))
        print("*************************************")
        # Compute confusion matrix
        y_test = test_Y
        y_pred = y_hat
        class_names = np.unique(y_pred)
        cnf_matrix = confusion_matrix(y_test, y_pred)
        np.set_printoptions(precision=2)
        # save confusion matrix to csv file
        df_confusion = pd.crosstab(y_test, y_pred)
        #df_confusion.to_csv(save_at+'confusion_matrix_for_vec_size_'+str(sizes[i])+'_sample_size_'+str(sample_size)+'.csv')
        if conf_mat_plot == 1:
            plot_conf_matrix_and_save(cnf_matrix,class_names,sizes[i],sample_size,save_at)
        
    f.close()
    
    #print ("Confusion matrix:\n", confusion_matrix(test_Y, y_hat))

if __name__ == "__main__":
    # define scope of the confusion plot
    rcParams.update({'figure.autolayout': True})
    # sample for each type. So total fragments = sample_size * file_type_count
    sample_size = [500,1000, 1500, 2000, 2500]
    # whether or not to plot confusion matrix in console and to .eps file
    conf_mat_plot = 0
    ratio = 0.7
    k = 3
    cross = 10
    for s in sample_size:
        main(s,ratio,k,cross,conf_mat_plot)