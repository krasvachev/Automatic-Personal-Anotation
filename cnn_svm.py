import numpy as np
#import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, InputLayer
from keras.utils import to_categorical
from keras import regularizers
from keras.models import Model
from sklearn.model_selection import StratifiedKFold, train_test_split
#from matplotlib import pyplot


class CNN_SVM:

    
    def __init__(self, parameters, num_classes, shape, preprocess_only = False):
        self.__data_loaded = False
        self.__model_trained = False
        self.__model_parameters = parameters
        self.__num_classes = num_classes
        self.__preprocess_only = preprocess_only
        self.__define_model(shape, parameters)

    def __define_model(self, shape, params):
        if(params['RGB'] == False):
            shape = (shape[0],shape[1],1)

        model = Sequential()
        model.add(InputLayer(input_shape=shape))
        for i in range(params['ConvCount']):
            cur_filt_size = params['ConvFilterSizes'][i]
            cur_filt_count = params['ConvFilterCount'][i]
            model.add(Conv2D(cur_filt_count, kernel_size=(cur_filt_size, cur_filt_size), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))   #1024
        if(self.__preprocess_only):
            model.add(Dense(768, activation='relu')) #128 256! 512!!! 768!!!!
            model.add(Dense(self.__num_classes, activation='softmax'))
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])   #loss: sparse_categorical_crossentropy oif there are a lot of classes metrics: mae, mse, msle(loss)
        else:
            model.add(Dense(self.__num_classes, activation='linear', kernel_regularizer=regularizers.l2(0.0001)))
            model.compile(loss='squared_hinge',optimizer='adam', metrics=['accuracy'])

        model.summary()
        #model.metrics_names
        
        self.__model = model

        
    def train_model(self, X, y, val_split = False):
        if(len(X.shape) < 4):
            X = X.reshape(X.shape[0],X.shape[1],X.shape[2],1)
        y_categ = to_categorical(y, self.__num_classes)


        if(val_split != False):
            X_tr, X_ts, y_tr_categ, y_ts_categ = train_test_split(X, y_categ, test_size=val_split, random_state=42)
            data_count = X_tr.shape[0]
            self.__model.fit(X_tr, y_tr_categ, batch_size=data_count, epochs=1,  verbose=1, validation_data=(X_ts, y_ts_categ))
        else:
            res = self.__model.train_on_batch(X, y_categ)
            #pyplot.plot(res.history['mean_squared_error'])
            print(res)
            #print("Accuracy = ", res[0], "Loss = ", res[1], "MSE = ", res[2])


    def evaluate_model(self, X):
        if(len(X.shape) < 4):
            X = X.reshape(X.shape[0],X.shape[1],X.shape[2],1)
        X = X.reshape(X.shape[0],X.shape[1],X.shape[2],1)

        return np.argmax(self.__model.predict(X), axis=-1)
        
    def get_feat_vec(self, new_data):
        old_model = self.__model
        sh = new_data.shape
        if(len(new_data.shape) < 4):
            new_data = new_data.reshape(sh[0],sh[1],sh[2],1)

        layer_count = len(old_model.layers)
        new_model = Model(inputs=old_model.input, outputs=old_model.layers[layer_count-2].output) #-2
        new_features = new_model.predict(new_data)
        return new_features

