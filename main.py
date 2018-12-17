# -*- coding: utf-8 -*-
"""
This is yossi and idan program to implement the computer vision first task.
"""
from CVPipelines.CVPipelineLinear import CVPipelineLinear
from CVPipelines.CVPipelineNonLinear import CVPipelineNonLinear
import time
import os


def main():

    fold1 = [x for x in range(0, 10)]
    class_indices = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    path_photos = "C:/Users/yossi/PycharmProjects/ComputerVisionTask1/101_ObjectCategories"
    path_results = os.getcwd() + "/results"

    pipeline_linear = CVPipelineLinear("dense SIFT + KMEANS + linear SVM", fold1=fold1, fold2=class_indices,
                                       path=path_photos, results=path_results)
    pipeline_non_linear = CVPipelineNonLinear("dense SIFT + KMEANS + RBF kernel SVM", fold1=fold1, fold2=class_indices,
                                              path=path_photos, kernel="rbf", results=path_results)
    start_time = time.time()

    start_time_model = time.time()

    print("linear model")
    pipeline_linear.get_data()
    print("time for get_data in seconds is {}".format((time.time() - start_time)))
    new_time = time.time()
    pipeline_linear.data_preprocess()
    print("time for prepare in seconds is {}".format((time.time() - new_time)))
    new_time = time.time()
    pipeline_linear.split_the_data()
    print("time for split_the_data in seconds is {}".format((time.time() - new_time)))
    new_time = time.time()
    pipeline_linear.train_with_tuning()
    print("time for train_with_tuning in seconds is {}".format((time.time() - new_time)))
    new_time = time.time()
    pipeline_linear.train()
    print("time for train in seconds is {}".format((time.time() - new_time)))
    new_time = time.time()
    pipeline_linear.test()
    print("time for test in seconds is {}".format((time.time() - new_time)))
    new_time = time.time()
    pipeline_linear.evaluate()
    print("time for evaluate in seconds is {}".format((time.time() - new_time)))
    new_time = time.time()
    pipeline_linear.report_results()
    print("time for report_results in seconds is {}".format((time.time() - new_time)))
    print("total time for the linear model in seconds is: {}".format(time.time() - start_time_model))
    print("")

    start_time_model = time.time()
    print("non linear model")
    pipeline_non_linear.get_data()
    print("time for get_data in seconds is {}".format((time.time() - start_time_model)))
    new_time = time.time()
    pipeline_non_linear.data_preprocess()
    print("time for prepare in seconds is {}".format((time.time() - new_time)))
    new_time = time.time()
    pipeline_non_linear.split_the_data()
    print("time for split_the_data in seconds is {}".format((time.time() - new_time)))
    new_time = time.time()
    pipeline_non_linear.train_with_tuning()
    print("time for train_with_tuning in seconds is {}".format((time.time() - new_time)))
    new_time = time.time()
    pipeline_non_linear.train()
    print("time for train in seconds is {}".format((time.time() - new_time)))
    new_time = time.time()
    pipeline_non_linear.test()
    print("time for test in seconds is {}".format((time.time() - new_time)))
    new_time = time.time()
    pipeline_non_linear.evaluate()
    print("time for evaluate in seconds is {}".format((time.time() - new_time)))
    new_time = time.time()
    pipeline_non_linear.report_results()
    print("time for report_results in seconds is {}".format((time.time() - new_time)))
    print("total time for the non linear model in seconds is: {}".format(time.time() - start_time_model))
    print("")

    print("the total time of both models in minutes is: {}".format((time.time() - start_time) / 60))

if __name__ == "__main__":

    main()

