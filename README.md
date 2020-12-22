# Automatic-Personal-Anotation

The implemented network is solving the problem mentioned above - the automatic personal annotation. Nowadays, the avaliable amount of data is large. Because of the social medias there are a dozens of images needed to be classified. A good question is how to organized and store them effectively. One of the ways to do that is to categorize the images in different classes. Unfortunately, when faces are classified, there are a lot of uncontrolled conditions like varying of poses, expressions, illumination, etc. 

For the task it is used an algorithm for detection and tracking the person implemented in Kinect SDK. Then it's developed an algorithm to obtain the face with the face orientation. The face images are saved and use for the neural network. The goal of this is the whole algorithm to be implemented for multi-view systems.

Here, only the implementation of the neural network is given. In data_load.py the data is loaded. It could be used CroppedYale DB or a DB generated from Kinect. The neural network is defined in cnn_svm.py. It's a combination of CNN and SVM. The main file is facerec.py. There the magic is done. Metrics, like confusion matrix and accuracy, are implemented there for validation of the neural network performance.


