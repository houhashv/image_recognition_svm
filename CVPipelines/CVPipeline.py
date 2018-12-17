# -*- coding: utf-8 -*-
"""
PyCharm Editor

This is yossi first file.

"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import os
import random
import cv2
import time
from sklearn.svm import LinearSVC
from CVPipelines.SVMNoneLinear import SVMNoneLinear
from sklearn.metrics import confusion_matrix
import pickle
import matplotlib.pyplot as plt


class CVPipeline:

    def __init__(self, pipeline_type, fold1=None, fold2=None, path=None, seed=0, num_of_classes=10, train_size=20,
                 results=None):

        self.pipeline_type = pipeline_type
        self.results_path = results
        self.kernel = None
        self.seed = seed

        self._min_s = 80
        self._max_x = 400
        self._x_step = 30

        self._min_clusters = 100
        self._max_clusters = 900
        self._clusters_step = 100

        self._num_of_classes = num_of_classes
        self._train_size = train_size

        self._images = None

        self._S_range = range(self._min_s, self._max_x, self._x_step)
        self._K_range = [x for x in range(self._min_clusters, self._max_clusters, self._clusters_step)]
        self._C_range = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
        self._sift_step_size_M_range = [5, 10]
        self._sift_scale_radii_range = [8, 16, 24, 32]
        self._gamma_range = ['auto']
        self._degree_range = [3]

        self._best_dummy = 1
        self._best_i_dummy = -1

        self.params = dict()
        self.params["path"] = path
        self.params["seed"] = seed
        self.params["hyper_parameters"] = {"S": {"range": self._S_range, "values": {x: [] for x in self._S_range},
                                                 "best": {"value": self._best_dummy, "i": self._best_i_dummy}},
                                           "K": {"range": self._K_range, "values": {x: [] for x in self._K_range},
                                                 "best": {"value": self._best_dummy, "i": self._best_i_dummy}},
                                           "C": {"range": self._C_range, "values": {x: [] for x in self._C_range},
                                                 "best": {"value": self._best_dummy, "i": self._best_i_dummy}},
                                           "sift_step_size_M": {"range": self._sift_step_size_M_range,
                                                                "values": {x: [] for x in self._sift_step_size_M_range},
                                                                "best": {"value": self._best_dummy,
                                                                         "i": self._best_i_dummy}},
                                           "sift_scale_radii": {"range": self._sift_scale_radii_range,
                                                                "values": {x: [] for x in self._sift_scale_radii_range},
                                                                "best": {"value": self._best_dummy,
                                                                         "i": self._best_i_dummy}},
                                           "gamma": {"range": self._gamma_range, "values": {x: [] for x in self._gamma_range},
                                                     "best": {"value": self._best_dummy, "i": self._best_i_dummy}},
                                           "degree": {"range": self._degree_range, "values": {x: [] for x in self._degree_range},
                                                      "best": {"value": self._best_dummy,"i": self._best_i_dummy}}
                                           }

        self.params["max_data_size"] = self._train_size * 2
        self.params["chance_to_use_sift"] = 0.5
        self.params["chance_to_use_image"] = 0.2
        self.results = dict()
        self.results["confusion_matrix"] = None

        self.dictionaries = {}

        sample_indices = random.sample(range(101), self._num_of_classes * 2)

        if fold1 is None and fold2 is None:

            fold1 = sample_indices[:self._num_of_classes]
            fold2 = sample_indices[self._num_of_classes:]

        elif fold1 is None:

            fold1 = sample_indices[:self._num_of_classes]

        elif fold2 is None:

            fold2 = sample_indices[:self._num_of_classes]

        self.params["fold1"] = fold1
        self.params["fold2"] = fold2

        self.fold1_dirs = None
        self.fold2_dirs = None
        self.indexes = {"fold1": {}, "fold2": {}}

        self.data = {"fold1": {"train": {"features": {}, "features_represented": {}, "labels": []},
                               "validation": {"features": {}, "features_represented": {}, "labels": []},
                               "full_data": {"features": [], "features_prepare": {}, "labels": []}},
                     "fold2": {"train": {"features": {}, "features_represented": {}, "labels": []},
                               "test": {"features": {}, "features_represented": {}, "labels": []},
                               "full_data": {"features": [], "features_prepare": {}, "labels": []}}}

        for s in self.params["hyper_parameters"]["S"]["range"]:
            self.data["fold1"]["train"]["features"][s] = []
            self.data["fold1"]["validation"]["features"][s] = []
            self.data["fold1"]["train"]["features_represented"][s] = []
            self.data["fold1"]["validation"]["features_represented"][s] = []
            self.data["fold1"]["full_data"]["features_prepare"][s] = []
            self.data["fold2"]["train"]["features"][s] = []
            self.data["fold2"]["test"]["features"][s] = []
            self.data["fold2"]["train"]["features_represented"][s] = []
            self.data["fold2"]["test"]["features_represented"][s] = []
            self.data["fold2"]["full_data"]["features_prepare"][s] = []

        self.best_error_tuning = 1
        self.error_test = 1
        self.model = None
        self.data_train_svm = None
        self.data_test_svm = None
        self.real_values_train = None
        self.real_values_test = None
        self.predictions = None
        self.confusion_matrix = None

    def get_data(self):
        """

        :return:
        """
        all_dirs = os.listdir(self.params["path"])
        self.fold1_dirs = [all_dirs[i] for i in self.params["fold1"]]
        self.fold2_dirs = [all_dirs[i] for i in self.params["fold2"]]

        for ind, fold in enumerate((self.fold1_dirs, self.fold2_dirs)):

            index = 0

            for image_class in fold:
                path_folder = self.params["path"] + "/" + image_class
                all_images = os.listdir(path_folder)
                folder_size = len(all_images)
                min_size = min(self.params["max_data_size"], folder_size)
                self.indexes["fold{}".format(ind + 1)][image_class] = [x for x in range(index, index + min_size)]
                index += min_size
                self._images = random.sample(range(folder_size), min_size)
                for i in self._images:
                    image_path = self.params["path"] + "/{}/".format(image_class) + all_images[i]
                    img = cv2.imread(image_path, 1)
                    self.data["fold{}".format(ind + 1)]["full_data"]["labels"].append(image_class)
                    self.data["fold{}".format(ind + 1)]["full_data"]["features"].append(img)

    def data_preprocess(self):
        """

        :return:
        """
        for ind, fold in enumerate((self.data["fold1"]["full_data"], self.data["fold2"]["full_data"])):
            for s in self.params["hyper_parameters"]["S"]["range"]:
                for img in fold["features"]:
                    img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img_resize = cv2.resize(img_g, (s, s))
                    self.data["fold{}".format(ind + 1)]["full_data"]["features_prepare"][s].append(img_resize)

    def split_the_data(self):
        """

        :return:
        """
        for i, fold in enumerate(self.indexes.values()):

            data = self.data["fold{}".format(i + 1)]
            for class_name, indexes in fold.items():
                for s in self.data["fold{}".format(i + 1)]["train"]["features"].keys():
                    data["train"]["features"][s] += \
                        data["full_data"]["features_prepare"][s][indexes[0]:indexes[self._train_size]]

                    data["validation" if i == 0 else "test"]["features"][s] += \
                        data["full_data"]["features_prepare"][s][indexes[self._train_size]:indexes[-1] + 1]

                data["train"]["labels"] += data["full_data"]["labels"][indexes[0]:indexes[self._train_size]]
                data["validation" if i == 0 else "test"]["labels"] += \
                    data["full_data"]["labels"][indexes[self._train_size]:indexes[-1] + 1]

    def get_default_parameters(self):
        """

        :return:
        """
        return self.params

    def _dense_sift(self, img, s, m, scale):
        """

        :param img:
        :param s:
        :param m:
        :param scale:
        :return:
        """
        sift = cv2.xfeatures2d.SIFT_create()

        kp = [cv2.KeyPoint(x, y, scale) for y in range(0, s, m)
              for x in range(0, s, m)]

        return sift.compute(img, kp)

    def _bag_of_words(self, data, s, k, scale, m):
        """

        :param data:
        :param s:
        :param k:
        :param scale:
        :param m:
        :return:
        """
        sample_sifts = None
        sample_images = [i for i in data if random.random() <= self.params["chance_to_use_image"]]

        for img in sample_images:

            dense_feat = self._dense_sift(img, s, m, scale)
            sample_size = int(dense_feat[1].shape[0] * self.params["chance_to_use_sift"])
            if sample_size == dense_feat[1].shape[0]:
                print(dense_feat[1].shape[0], data, s, k, scale, m)

            sample_indexes = random.sample(range(0, dense_feat[1].shape[0]), sample_size)

            if sample_sifts is None:
                sample_sifts = dense_feat[1][sample_indexes]
            else:
                sample_sifts = np.append(sample_sifts, dense_feat[1][sample_indexes], axis=0)

        kmeans = KMeans(n_clusters=k)
        kmeans.fit(sample_sifts)
        self.dictionaries[s] = kmeans

    def represent(self, data=None, s=None, k=None, scale=None, m=None, fold=None, dataset=None):
        """

        :param data:
        :param s:
        :param k:
        :param scale:
        :param m:
        :param fold:
        :param dataset:
        :return:
        """
        if data is None:

            s = self.params["hyper_parameters"]["S"]["best"]["value"]
            k = self.params["hyper_parameters"]["K"]["best"]["value"]
            m = self.params["hyper_parameters"]["sift_step_size_M"]["best"]["value"]
            scale = self.params["hyper_parameters"]["sift_scale_radii"]["best"]["value"]
            data = self.data[fold][dataset]["features"][s]

        self.data[fold][dataset]["features_represented"][s] = []

        for img in data:

            dense_feat_dict = self._dense_sift(img, s, m, scale)
            image_sifts = dense_feat_dict[1]
            labels = self.dictionaries[s].predict(image_sifts)
            uniques = [x for x in range(0, k)]
            labels_counts = [[x, labels.tolist().count(x)] for x in uniques]
            labels_count = pd.DataFrame(labels_counts, columns=["label", "count"])
            labels_normalized = labels_count["count"] / labels_count["count"].sum()
            self.data[fold][dataset]["features_represented"][s].append(labels_normalized.tolist())

    def _prepare_data_for_svm(self, s, k, m, scale, fold):
        """

        :param s:
        :param k:
        :param m:
        :param scale:
        :param fold:
        :return:
        """
        data_train_kmeans = self.data[fold]["train"]["features"][s]
        data_test_kmeans = self.data[fold]["validation" if fold == "fold1" else "test"]["features"][s]
        self._bag_of_words(data_train_kmeans, s, k, scale, m)
        self.represent(data_train_kmeans, s, k, scale, m, fold, "train")
        self.represent(data_test_kmeans, s, k, scale, m, fold, "validation" if fold == "fold1" else "test")
        real_values_train = self.data[fold]["train"]["labels"]
        real_values_test = self.data[fold]["validation" if fold == "fold1" else "test"]["labels"]
        data_train_svm = self.data[fold]["train"]["features_represented"][s]
        data_test_svm = self.data[fold]["validation" if fold == "fold1" else "test"]["features_represented"][s]

        return data_train_svm, data_test_svm, real_values_train, real_values_test

    def svm_c(self, c, gamma, degree):
        """

        :param c:
        :param gamma:
        :param degree:
        :return:
        """
        if self.kernel == "linear":
            return LinearSVC(C=c)
        else:
            return SVMNoneLinear(kernel=self.kernel, C=c, degree=degree, gamma=gamma)

    def _best_update(self, error, **kwargs):
        """

        :param error:
        :param kwargs:
        :return:
        """
        self.best_error_tuning = error

        for hyper, values in kwargs.items():

            self.params["hyper_parameters"][hyper]["best"]["value"] = values["value"]
            self.params["hyper_parameters"][hyper]["best"]["i"] = values["index"]
            self.params["hyper_parameters"][hyper]["values"][values["value"]].append(error)

    def train_with_tuning(self):
        """

        :return:
        """
        iteration = 0
        start_time = time.time()
        stats = []

        for m_i, m in enumerate(self.params["hyper_parameters"]["sift_step_size_M"]["range"]):

            for scale_i, scale in enumerate(self.params["hyper_parameters"]["sift_scale_radii"]["range"]):

                for s_i, s in enumerate(self.params["hyper_parameters"]["S"]["range"]):

                    for k_i, k in enumerate(self.params["hyper_parameters"]["K"]["range"]):

                        data_train_svm, data_test_svm, real_values_train, real_values_test = \
                            self._prepare_data_for_svm(s, k, m, scale, "fold1")

                        for c_i, c in enumerate(self.params["hyper_parameters"]["C"]["range"]):

                            for d_i, degree in enumerate(self.params["hyper_parameters"]["degree"]["range"]):

                                for g_i, gamma in enumerate(self.params["hyper_parameters"]["gamma"]["range"]):

                                    svm = self.svm_c(c, gamma, degree)
                                    svm.fit(data_train_svm, real_values_train)
                                    # predictions = svm.predict(data_test_svm)
                                    error = 1 - svm.score(data_test_svm, real_values_test)

                                    if error < self.best_error_tuning:

                                        self._best_update(error, S={"value": s, "index": s_i},
                                                          C={"value": c, "index": c_i},
                                                          K={"value": k, "index": k_i},
                                                          sift_scale_radii={"value": scale, "index": scale_i},
                                                          sift_step_size_M={"value": m, "index": m_i},
                                                          gamma={"value": gamma, "index": g_i},
                                                          degree={"value": degree, "index": d_i})

                                    timing = (time.time() - start_time) / 60
                                    print("time for iteration {} in minutes: {}".format(iteration, timing))
                                    print("error for iteration {} is: {}".format(iteration, error))
                                    print("hyper params for iteration {} are: S:{}, K:{}, C:{}, M:{}, radii:{},"
                                          .format(iteration, s, k, c, m, scale) +
                                          " gamma: {}, degree: {}".format(gamma, degree))

                                    iteration += 1
                                    start_time = time.time()
                                    stats.append({"iteration": iteration, "time": timing, "error": error,
                                                  "S": s, "K": k, "C": c, "M": m, "radii": scale, "gamma": gamma,
                                                  "degree": degree})

        pickle.dump(stats, open("{}/stats_{}.p".format(self.results_path, self.kernel), "wb"))
        pickle.dump(self.params["hyper_parameters"], open("{}/hyper_params_{}.p".format(self.results_path, self.kernel),
                                                          "wb"))

    def train(self):
        """

        :return:
        """
        self.params["hyper_parameters"] = pickle.load(open("{}/hyper_params_{}.p".format(self.results_path, self.kernel),
                                                           'rb'))

        s = self.params["hyper_parameters"]["S"]["best"]["value"]
        k = self.params["hyper_parameters"]["K"]["best"]["value"]
        c = self.params["hyper_parameters"]["C"]["best"]["value"]
        m = self.params["hyper_parameters"]["sift_step_size_M"]["best"]["value"]
        scale = self.params["hyper_parameters"]["sift_scale_radii"]["best"]["value"]
        gamma = self.params["hyper_parameters"]["gamma"]["best"]["value"]
        degree = self.params["hyper_parameters"]["degree"]["best"]["value"]

        self.data_train_svm, self.data_test_svm, self.real_values_train, self.real_values_test = \
            self._prepare_data_for_svm(s, k, m, scale, "fold2")

        svm = self.svm_c(c, gamma, degree)
        svm.fit(self.data_train_svm, self.real_values_train)
        self.model = svm

    def test(self):
        """

        :return:
        """
        self.predictions = self.model.predict(self.data_test_svm)
        self.error_test = 1 - self.model.score(self.data_test_svm, self.real_values_test)

    def evaluate(self):
        """

        :return:
        """
        labels = list(set(self.real_values_test)).sort()
        self.confusion_matrix = confusion_matrix(self.real_values_test, self.predictions, labels=labels)

    def _largest_margin(self):
        """

        :return:
        """
        classes = set(self.real_values_test)
        print("classes in test: {}".format(classes))
        classes_check = []
        for class_name in classes:
            if class_name not in classes_check:

                classes_check.append(class_name)
                samples_i = [(np.array(self.data_test_svm[i]).reshape(1, -1), self.real_values_test[i]) if self.real_values_test[i] == class_name
                             else None for i, value in enumerate(self.real_values_test)]
                samples_i = [x for x in samples_i if x is not None]
                margin_errors = []

                for i, sample in enumerate(samples_i):

                    if self.model.predict(sample[0]) != sample[1]:

                        scores = self.model.decision_function(sample[0])
                        scores = scores if self.kernel != "linear" else scores[0]
                        correct_index = self.fold2_dirs.index(sample[1])
                        distance = scores[correct_index] - scores[np.argmax(scores)]
                        margin_errors.append((i, distance))

                margin_errors_values = [x[1] for x in margin_errors]
                margin_errors_values = sorted([x for x in margin_errors_values], reverse=True)\
                    [:min(2, len(margin_errors_values))]
                margin_errors = [x for x in margin_errors if x[1] in margin_errors_values]

                for i in margin_errors:
                    path_folder = self.params["path"] + "/" + class_name
                    all_images = os.listdir(path_folder)
                    image_path = self.params["path"] + "/{}/".format(class_name) + all_images[20 + i[0]]
                    mg = cv2.imread(image_path, -1)
                    image_string = 'image_{}_{}'.format(class_name, all_images[20 + i[0]])
                    cv2.imshow(image_string, mg)
                    print("error for: {}, is: {}".format(image_string, i[1]))

    def plot_hyper_parameters(self):
        """

        :return:
        """
        stats_linear = pickle.load(open(os.getcwd() + "/results/stats_{}.p".format(self.kernel), "rb"))
        df = pd.DataFrame(stats_linear).rename({"M": "sift_step_size_M", "radii": "sift_scale_radii"}, axis='columns')
        best_hyper_parameter = pickle.load(open(os.getcwd() + "/results/hyper_params_{}.p".format(self.kernel), "rb"))
        exclude = ["error", "iteration", "time"]
        exclude2 = exclude
        if self.kernel == "linear":
            exclude2 = exclude
            exclude2 += ["gamma", "degree"]
        else:
            exclude2 += ["degree"]

        keys = [key for key in best_hyper_parameter.keys() if key not in exclude2]
        count = len(keys)

        for key in keys:
            keys_next = [key1 for key1 in best_hyper_parameter.keys() if key1 != key and key not in exclude]
            ranges = df.loc[(df[keys_next[0]] == best_hyper_parameter[keys_next[0]]["best"]["value"]) &
                            (df[keys_next[1]] == best_hyper_parameter[keys_next[1]]["best"]["value"]) &
                            (df[keys_next[2]] == best_hyper_parameter[keys_next[2]]["best"]["value"]) &
                            (df[keys_next[3]] == best_hyper_parameter[keys_next[3]]["best"]["value"]) &
                            (df[keys_next[4]] == best_hyper_parameter[keys_next[4]]["best"]["value"]) &
                            (df[keys_next[5]] == best_hyper_parameter[keys_next[5]]["best"]["value"]),
                            [key, 'error']]
            fig = plt.figure()
            fig.add_axes((.1, .4, .8, .5))
            ranges[key] = ranges[key].apply(lambda x: x if type(x) != type("") else -1)
            plt.plot(ranges[key], ranges["error"])
            plt.title('Validation Error VS. {}'.format(key))
            plt.xlabel(key)
            plt.ylabel('Validation Error')
            len_a = len(best_hyper_parameter[key]["range"]) - 1
            steps = len(best_hyper_parameter[key]["range"])
            range_0 = best_hyper_parameter[key]["range"][0]
            range_last = best_hyper_parameter[key]["range"][len_a]
            best_value = best_hyper_parameter[key]["best"]["value"]
            text = '{} was changed in {} pixels steps from range of {} to {}\n'.format(key, steps, range_0, range_last)
            text += 'The {} that got the best Validation error was {}'.format(key, best_value)
            fig.text(.1, .1, text)
            count -= 1
            plt.show(block=True if count == 0 else False)

    def report_results(self):
        """

        :return:
        """
        print("the error on the test dataset is:{}".format(self.error_test))
        print("the confusion_matrix over the test dataset is: \n{}".format(self.confusion_matrix))
        self._largest_margin()
        self.plot_hyper_parameters()
