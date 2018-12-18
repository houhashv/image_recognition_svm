# Image classifier using Kmenas, dense SIFTs and Linear and Non linear SVM

Implement an object recognition pipe
Train it on a subset of the Caletch101 dataset
Including tuning of hyper parameters
Test your accuracy on another subset
Analyze and report your results

We will use publically available implementations for all of the algorithms, and the main programming effort is composing the code into a functional pipe and experimenting with it to get reasonable results.
The exercise goals: 
	Experience building a full pipe including representation. 
	Get acquainted with classic representation techniques. 
Data set

We will work with Caltech 101 dataset
	The data contains 101 classes, with 31-800 images each.
	Data homepage: http://www.vision.caltech.edu/Image_Datasets/Caltech101/
	We will train on 20 images per class , test on 20 others (unless there are less than 40 images for the class – in which case we will  
have less images for test) 
	When loading the data:
	The data is in 102 folders in the top folder 101_ObjectCategories\, one folder per class
	Note that the folder  BACKGROUND_Google\ is not a class. Erase it.
	To load it write a function which uses the function listdir() (os package) on the top folder 101_ObjectCategories\, then loops through the subfolders and through the images in each subfolder. At the end of the process you should have the images in a single 3D Data array of size S×S×N (with S a side length parameter, and N the number of images) and a single Labels vector of size 1×N
	Load images using imread() (cv2 package)
	Get them to gray scale using rgb2gray() (cv2 package)
	Resize them to S×S using  imresize()  (cv2 package) and stack them into the data array
	Highly recommended – to save time in your work, once you load the data, save it as a .pkl file (pickle package). This way, instead of loading the images from scratch each time you run (which may take some time), you can load the .pkl file. This is way faster. For more information see https://docs.python.org/2/library/pickle.html
	Data split 
	Debug and tune your pipe on the first 10 classes (fold 1).
	When the final configuration is debugged and stable, run the algorithm with the best hyper parameter configuration found on classes 11-20 (fold 2). 

## Required packages:

General packages: os, numpy, pickle, matplotlib, cv2
SVM algorithm: sklearn (link). 
SIFT algorithm: cv2 (link) to sift description. For dense sift, see dense_sift_example.py). In order to run dense SIFT, make sure you install version 3.4.2.16 of cv2 and its expansion. To do so, run the following commands in the terminal:
pip install opencv-python==3.4.2.16
pip install opencv-contrib-python==3.4.2.16
Kmeans algorithm: sklearn (link)

## Important details for algorithms we use

### SVM algorithm: 
Important parameters - C the SVM tradeoff parameter, gamma when using an RBF kernel, degree the polynomial degree when using a polynomial kernel. 
In order implement "one vs. all" approach with a linear case of SVM, use "LinearSVC" function and set the variable multi_class=”ovr”. For other cases, there is no implementation by sklearn, and you'll need to implement it:
	Write a function n_class_SVM_train which loops through the M classes and trains M binary classifiers in one-versus-all method
	That is: for classifier i, the examples of class i get the label +1, and the rest of the classes get -1 label.
	The function returns an n_class_SVM array containing all the n_class SVM models that have the shape [n_class,n_features].
	Write an n_class_SVM_predict() function which accepts a set of examples to test as a N*n_features matrix, and an MClassSVM structure returned by MClassSVM_Train. The function predicts by
	Apply the n_class SVMs to the data and put the predictions in an N×n_class class score matrix. 
	Compute the predicted class (an N×1 vector) by taking the argmax over the class score matrix columns (the highest score in the row determines the winning class) 
	Return the predicted class and the class score matrix

### SIFT algorithm: 
We extract SIFTs around every point in a dense 2D grid of points over the image. The grid stride\step (i.e. the distance between points) is likely to be an important parameter. Also, at each point SIFTS may be extracted at multiple scales. I suggest using the scales used at http://vision.cse.psu.edu/seminars/talks/2009/random_tff/bosch07a.pdf, section 5,’Appearance’. This is a good starting point.
Kmeans algorithm: 
When training the dictionary, it is enough to extract some SIFTS from some images (that is: you do not have to use all the SIFTS from all the images).  For example, using a subset of 100 SIFTS from 1000 images (100,000 SIFTs overall) should be enough for good clustering). The most Important Parameter is K (n_clusters). Several hundred is a good area to look in.

## Required submission

## Code:
The code is runnig directly, without modifications, on any machine. 
The only modification in the code will be the path to the dataset folder.
This machine will gor to have all the required packages. This path should appear at the beginning of the code in a variable named – “data_path”.
The code will be generic and will be able to run on any subset of 10 classes (from the 101 classes) by changing a single variable called class_indices which will be defined at the beginning of the main file, in one of the first 5 lines. class_indices will be a vector of 1*10 containing the indices of classes on which the experiment will run. For example:
class_indices = [ 5 6 10 60 65 67 81 83 86 90]
Important: The code you submit should set class_indices to "fold 2" classes, i.e.  class_indices = [11:20]. It should use fixed hyper parameters (the best configuration of hyper-parameters tested, found on "fold 1", i.e. classes [1:10]). However, I will test your code on an arbitrary set of classes. You should verify that it is able to run on such an arbitrary set.
	The code should print to the python run as output the test error result (in a clear sentence) and the confusion matrix.
## Report:
The report should include
	Hyper parameter tuning graphs (use matplotlib package for plotting):  The pipe includes hyper parameters (like S – the image size,  K – the number of codewords for Kmeans, C – the SVM tradeoff parameter, etc..).  Such parameters should be tuned to get good accuracy.  I list below the most important hyper parameters for each module, but you may consider tuning other parameters. 
	For at least two parameters, systematic tuning should be done, in which:
	The training set is split to two subsets, termed 'training' and 'validation'
	Several values of the parameter are tested, by training on the training set and estimate the error on the validation set.
	The parameter value giving the lowest validation error is chosen.
	The report should include graphs showing the validation error as a function of hyper parameter value for the hyper parameters that were systematically tuned. The chosen value should be stated.
	Each graph should include a coherent and informative caption.
	For each tuned hyper-parameter, a short explanation of the parameter should be included.

### Test results:
The error rate obtained over the test set.
A confusion matrix (use sklearn.metrices.confusion_matrix(…)) and its analysis.

### Error visualization: 
For each class, show images of the two largest errors on images of the class (i.e. images from the class which were miss-classified). The error images should be displayed only if they exist (i.e. if there are at least two errors from the class). If there is only one error from the class –show it, and if there are none – just state that there were no errors for this class.  By largest error I mean the images which got the lowest margin, following this definition:
Class_score(i): For SVM-based system Class_score(i) is defined as the SVM score of the i-th classifier. 
The margin for an example of class i is Class_score(i)-max┬(j≠i)⁡〖Class_score(j)〗. This is the difference between the score of the correct class score and the maximal score of incorrect classes. Larger values indicate higher confidence. A value below 0 is an error.

