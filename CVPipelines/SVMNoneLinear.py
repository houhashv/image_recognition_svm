from sklearn.svm import SVC
import numpy as np


class SVMNoneLinear:

    def __init__(self, C=0.001, kernel="rbf", degree=3, gamma='auto', decision_function_shape='ovr'):
        '''
        :param c: the penalty parameter C of the error term
        :param kernel: the kernel to use (in our case is the rbf only)
        :param degree: the degree of the polynomial (not used)
        :param gamma: the kernel coefficient for ‘rbf’
        :param decision_function_shape: the
        '''
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        # self.decision_function_shape = decision_function_shape
        # the classifiers to use
        self.classifiers = {}

    def fit(self, features, y):
        '''
        fiting the algorithm over the one verses rest method
        :param features: the features from the image
        :param y: the class to predict
        :return:
        '''
        for class_name in set(y):

            svm = SVC(kernel=self.kernel, C=self.C, degree=self.degree, gamma=self.gamma)
            self.classifiers[class_name] = svm
            y_class = [label if label == class_name else "other" for label in y]
            self.classifiers[class_name] = svm.fit(features, y_class)

    def predict(self, features):
        '''
        predicting the class
        :param features: the features of the image
        :return:
        '''
        predictions = []

        for sample in features:

            i_values = []
            classes = []

            for class_name, cls in self.classifiers.items():

                classes.append(class_name)
                i_values.append(cls.decision_function(np.array(sample).reshape(1, -1)))

            predictions.append(classes[np.argmin(i_values)])

        return np.asarray(predictions)

    def score(self, features, y):
        '''
        the socre of the algorithm
        :param features: the features of the image
        :param y: the class of the image
        :return:
        '''
        predictions = self.predict(features)

        correct = sum([1 if prediction == y[i] else 0 for i, prediction in enumerate(predictions)])
        score = correct / predictions.shape[0]

        return score

    def decision_function(self, features):
        '''
        the distance of the samples X to the separating hyperplane
        :param features: the features of the image
        :return:
        '''
        predictions = []

        for sample in features:

            i_values = []
            classes = list(self.classifiers.keys())
            classes.sort()

            for class_name in classes:
                cls = self.classifiers[class_name]
                i_values.append(-cls.decision_function(np.array(sample).reshape(1, -1)))

            # predictions.append(classes)

        # return np.asarray(predictions)

        return i_values