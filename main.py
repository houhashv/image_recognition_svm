# -*- coding: utf-8 -*-
"""
PyCharm Editor

This is yossi first file.

"""
import numpy as np
from CVPipelines.CVPipelineLinear import CVPipelineLinear
from CVPipelines.CVPipelineNonLinear import CVPipelineNonLinear
import time
import os
import pickle
import pandas as pd


def main():

    fold1 = [x for x in range(0, 10)]
    class_indices = [51, 48, 80, 54, 37, 86, 98, 32, 90, 38]
    path_photos = "C:/Users/yossi/Downloads/101_ObjectCategories"
    path_results = os.getcwd() + "/results"

    pipeline_linear = CVPipelineLinear("dense SIFT + KMEANS + linear SVM", fold1=fold1, fold2=class_indices,
                                       path=path_photos, results=path_results)
    pipeline_non_linear = CVPipelineNonLinear("dense SIFT + KMEANS + RBF kernel SVM", fold1=fold1, fold2=class_indices,
                                              path=path_photos, kernel="rbf", results=path_results)
    start_time = time.time()

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

    print("")
    start_time = time.time()
    print("non linear model")
    pipeline_non_linear.get_data()
    print("time for get_data in seconds is {}".format((time.time() - start_time)))
    new_time = time.time()
    pipeline_non_linear.data_preprocess()
    print("time for prepare in seconds is {}".format((time.time() - new_time)))
    new_time = time.time()
    pipeline_non_linear.split_the_data()
    print("time for split_the_data in seconds is {}".format((time.time() - new_time)))
    # new_time = time.time()
    # pipeline_non_linear.train_with_tuning()
    # print("time for train_with_tuning in seconds is {}".format((time.time() - new_time)))
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

    print("the total time in minutes is {}".format((time.time() - start_time) / 60))

if __name__ == "__main__":

    # main()
    hyper_params_linear = pickle.load(open("results/hyper_params_linear.p", "rb"))
    stats_linear = pickle.load(open("results/stats_linear.p", "rb"))
    df = pd.DataFrame(stats_linear)
    print(df[df["error"] == min(df["error"])])
    df.shape

    # stats_linear = pickle.load(open(os.getcwd() + "/CVPipelines/results/stats_linear.p", "rb"))
    # df = pd.DataFrame(stats_linear)
    # bestHyperParameter = df[df["error"] == min(df["error"])].iloc[0]

    # hyper_params_linear["S"]["best"]["value"] = bestHyperParameter["S"]
    # hyper_params_linear["S"]["best"]["i"] = df[df["error"] == min(df["error"])].iloc[0].name
    # hyper_params_linear["K"]["best"]["value"] = bestHyperParameter["K"]
    # hyper_params_linear["K"]["best"]["i"] = df[df["error"] == min(df["error"])].iloc[0].name
    # hyper_params_linear["C"]["best"]["value"] = bestHyperParameter["C"]
    # hyper_params_linear["C"]["best"]["i"] = df[df["error"] == min(df["error"])].iloc[0].name
    # hyper_params_linear["sift_step_size_M"]["best"]["value"] = bestHyperParameter["M"]
    # hyper_params_linear["sift_step_size_M"]["best"]["i"] = df[df["error"] == min(df["error"])].iloc[0].name
    # hyper_params_linear["sift_scale_radii"]["best"]["value"] = bestHyperParameter["radii"]
    # hyper_params_linear["sift_scale_radii"]["best"]["i"] = df[df["error"] == min(df["error"])].iloc[0].name
    # hyper_params_linear["gamma"]["best"]["value"] = bestHyperParameter["gamma"]
    # hyper_params_linear["gamma"]["best"]["i"] = df[df["error"] == min(df["error"])].iloc[0].name
    # hyper_params_linear["degree"]["best"]["value"] = bestHyperParameter["degree"]
    # hyper_params_linear["degree"]["best"]["i"] = df[df["error"] == min(df["error"])].iloc[0].name
    # pickle.dump(hyper_params_linear, open(os.getcwd() + "/CVPipelines/results/hyper_params_linear.p", "wb"))
