from sklearn.svm import SVC
import numpy as np


class SVMNoneLinear:

    def __init__(self, C=0.001, kernel="rbf", degree=3, gamma='auto', decision_function_shape='ovr'):
        '''

        :param c:
        :param kernel:
        :param degree:
        :param gamma:
        :param decision_function_shape:
        '''
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.decision_function_shape = decision_function_shape
        self.classifiers = {}

    def fit(self, features, y):
        '''

        :param features:
        :param y:
        :return:
        '''
        for class_name in set(y):

            svm = SVC(kernel=self.kernel, C=self.C, degree=self.degree, gamma=self.gamma)
            self.classifiers[class_name] = svm
            y_class = [label if label == class_name else "other" for label in y]
            self.classifiers[class_name] = svm.fit(features, y_class)

    def predict(self, features):
        '''

        :param features:
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

        :param features:
        :param y:
        :return:
        '''
        predictions = self.predict(features)

        correct = sum([1 if prediction == y[i] else 0 for i, prediction in enumerate(predictions)])
        score = correct / predictions.shape[0]

        return score

    def decision_function(self, features):
        '''

        :param features:
        :return:
        '''
        predictions = []

        for sample in features:

            i_values = []
            classes = []

            for class_name, cls in self.classifiers.items():

                classes.append(class_name)
                i_values.append(cls.decision_function(np.array(sample).reshape(1, -1)))

            predictions.append(classes)

        return np.asarray(predictions)
