from CVPipelines.CVPipeline import CVPipeline
from CVPipelines.SVMNoneLinear import SVMNoneLinear


class CVPipelineNonLinear(CVPipeline):

    def __init__(self, pipeline_type, fold1=None, fold2=None, kernel="rbf", path=None, seed=0,
                 num_of_classes=10, train_size=20, results=None):

        super(CVPipelineNonLinear, self).__init__(pipeline_type, fold1, fold2, path=path, seed=seed,
                                                  num_of_classes=num_of_classes, train_size=train_size, results=results)
        self.kernel = kernel
        self._gamma_range = ['auto', 0.001, 0.01, 0.1, 1]
        # self._gamma_range = ['auto', 0.001]
        self.params["hyper_parameters"]["gamma"] = {"range": self._gamma_range, "values": {x: [] for x in self._gamma_range},
                                                    "best": {"value": self._best_dummy, "i": self._best_i_dummy}}
