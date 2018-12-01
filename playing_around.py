# -*- coding: utf-8 -*-
"""
PyCharm Editor

This is yossi first file.

"""
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split


def get_default_parameters():

    params_default = dict()
    params_default["seed"] = 0

    return params_default


def set_seed(seed=0):

    np.randon.seed(seed)


def get_data(data=None):
    pass


def split_the_data(features, labels, split):

    splited_data = {"train": {}, "test": {}}
    splited_data["train"]["features"], splited_data["train"]["labels"], \
        splited_data["test"]["features"], splited_data["test"]["labels"] = train_test_split(features, labels, split)

def prepare(train, params):
    pass


def train(trainDataRep):
    pass

def train_with_tuning(model):
    pass


def test(model, testDataRep):
    pass


def evaluate(results, labels, summary):
    pass


def report_results(summary, report):
    pass

params = get_default_parameters()#(experiment specific parameters override)

set_seed(params["seed"])

dan_dl = get_data(params["data"])

split_data = split_the_data(dan_dl["features"],dan_dl["labels"],params["split"])

train_data_rep = prepare(split_data["train"]["features"],params["prepare"])
model =  train(train_data_rep)
model = train_with_tuning(model)
test_data_rep = prepare(split_data["test"]["data"], params["preapare"])
results = test(model, test_data_rep)

summary = evaluate(results, split_data["test"]["labels"], params["summary"])
report_results(summary, params["report"])

