import os
from sklearn import svm
#from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, roc_curve, auc, roc_auc_score
from data_load import DataLoader, Dataset
from pca2d import PCA2D
from cnn_svm import CNN_SVM
import numpy as np
import cv2 as cv2
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn

from sklearn import metrics
sn.set(style='white', context='notebook', palette='deep')


cur_dataset = Dataset.TU
#cur_dataset = Dataset.YALE_EX_CROPPED           #Yale Cropped
dataloader = DataLoader(cur_dataset, {'angle_limit':10, 'img_format':None})
X,y,z,v = dataloader.load_data(reload=False)     
#X = X / 255

plt.figure(1)
g = sn.countplot(y)    

#z is the number of classes
#v is the map of labels to which 0-num_classes correspond

num_val_splits = 10  
labels = list(range(0,z))


#CNN + SVM 10-fold crossvalidation
X_shape =  (X.shape[1], X.shape[2], X.shape[3])
#X_shape =  (X.shape[0], X.shape[1], X.shape[2])     # Yale Cropped

labels = list(range(0,z))
conf_mat = np.zeros((z,z))
num_epochs = 1
batch_count = 30 

accuracy_mean_temp = 0
precision_mean_temp = 0
recall_mean_temp = 0

skf = StratifiedKFold(n_splits=num_val_splits, shuffle=True)
i = 0
for train_split, test_split in skf.split(X, y):
    # Default: filter size [3,3,3], Filter count [8, 12, 16]
    # There a lot of variations the best ones: - [3, 3, 3] - [12, 16, 20], [8, 16, 32], [16, 32, 64]
    #                                          - [5, 3, 3] - [12, 16, 20 !!!], [16, 20, 24 !!!]
    #                                          - [5, 5, 3] - [8, 16, 32], [16, 20, 24]
    #                                          - [5, 5, 5] - [12, 16, 20], [16, 20, 24]
    cnn_s = CNN_SVM({'RGB':True,'ConvCount': 3, 'ConvFilterSizes':[5,3,3], 'ConvFilterCount':[12,16,20]}, z, X_shape, True)

    for f in [ f for f in os.listdir('temp_data/')]:
        os.remove(os.path.join('temp_data/', f))

    with open('temp_data/' + cur_dataset.name + '_train_split_' + str(i), 'wb') as f:
        X_train, y_train = X[train_split], y[train_split]
        np.savez(f, X_train, y_train)

    with open('temp_data/' + cur_dataset.name + '_test_split_' + str(i), 'wb') as f:
        X_test, y_test = X[test_split], y[test_split]
        np.savez(f, X_test, y_test)
 
    for e in range(0,num_epochs):

        with open('temp_data/' + cur_dataset.name + '_train_split_' + str(i), 'rb') as f:
            rdy_arr = np.load(f, allow_pickle=True)
            X_train, y_train = rdy_arr['arr_0'], rdy_arr['arr_1']

            rand_ind = np.array(list(range(0,X_train.shape[0])))
            np.random.shuffle(rand_ind)

            X_train = X_train[rand_ind, : , :]
            y_train = y_train[rand_ind]

            skf = KFold(n_splits=batch_count, shuffle=False)

            s = 0
            for _, part_ind in skf.split(X_train, y_train):
                with open('temp_data/' + cur_dataset.name + '_' + str(s), 'wb') as f:
                    X_part = X_train[part_ind]
                    y_part = y_train[part_ind]
                    np.savez(f, X_part, y_part)
                s+=1

        for j in range(0, batch_count):
            with open('temp_data/' + cur_dataset.name + '_' + str(j), 'rb') as f:
                rdy_arr = np.load(f, allow_pickle=True)
                X_batch, y_batch = rdy_arr['arr_0'], rdy_arr['arr_1']
                cnn_s.train_model(X_batch, y_batch, False)
                print("batch {0}/{1}".format(j+1,batch_count))
        print("epoch {0}/{1}".format(e+1,num_epochs))

    feat_vec_X_train = None
    y_train_svm = None

    with open('temp_data/' + cur_dataset.name + '_train_split_' + str(i), 'rb') as f:
        rdy_arr = np.load(f, allow_pickle=True)
        X_train, y_train_svm = rdy_arr['arr_0'], rdy_arr['arr_1']
        feat_vec_X_train = cnn_s.get_feat_vec(X_train)

    with open('temp_data/' + cur_dataset.name + '_test_split_' + str(i), 'rb') as f:
        rdy_arr = np.load(f, allow_pickle=True)
        X_test, y_test = rdy_arr['arr_0'], rdy_arr['arr_1'] 
        feat_vec_X_test = cnn_s.get_feat_vec(X_test)
       
        
        clf = svm.SVC(kernel='linear', C=0.17) #linear optimal parameter is C=0.17, for 4 epochs; Test for C=0.13, 0.15, 0.20
        #clf = svm.SVC(kernel='poly', C=1, degree=1.44, gamma=0.5) # Poly tests: degree 1.40, 1.44, 1.90
        #clf = svm.SVC(kernel='rbf', C=1, gamma=0.0005)        # 0.00065, 0.00066
        
        
        #clf = svm.SVC(kernel='sigmoid', C=1, gamma=6.8)        # Doesn't work well for 2 classes
        #clf = svm.SVC(kernel='poly', C=0.1, degree=1.5, gamma=6.8) #optimal parameters poly Plamen gamma
        
        clf.fit(feat_vec_X_train, y_train_svm)
        y_pred = clf.predict(feat_vec_X_test)
        
        y_train_score = clf.decision_function(feat_vec_X_train)
        
        #y_train_score_temp = roc_auc_score(y_train_svm, y_pred)
        
        false_pos_rate, true_pos_rate, thresholds = roc_curve(y_train_svm, y_train_score)
        roc_auc3 = auc(false_pos_rate, true_pos_rate)
        
        print(accuracy_score(y_test,y_pred))
        
        print("acuracy:", metrics.accuracy_score(y_test,y_pred))
        #precision score
        print("precision:", metrics.precision_score(y_test,y_pred))
        #recall score
        print("recall:", metrics.recall_score(y_test,y_pred))
        #print(metrics.classification_report(y_test, y_pred))
        
        accuracy_mean_temp += metrics.accuracy_score(y_test,y_pred) 
        precision_mean_temp += metrics.precision_score(y_test,y_pred)
        recall_mean_temp += metrics.recall_score(y_test,y_pred)
        
        res_c = confusion_matrix(y_test, y_pred, labels=labels)
        conf_mat += res_c

    for f in [ f for f in os.listdir('temp_data/')]:
        os.remove(os.path.join('temp_data/', f))

    i+=1

print("")

accuracy_mean = accuracy_mean_temp / num_val_splits
print("Mean accuracy: ", accuracy_mean)

precision_mean = precision_mean_temp / num_val_splits
print("Mean precision: ", precision_mean)

recall_mean = recall_mean_temp / num_val_splits
print("Mean recall: ", recall_mean)

plt.figure(2)
sn.heatmap(conf_mat, annot=True, annot_kws={"size": 10})
#sn.heatmap(conf_mat/np.sum(conf_mat), annot=True, annot_kws={"size": 10})

# plt.figure(3)
# plt.scatter(false_pos_rate, true_pos_rate)

plt.show()

