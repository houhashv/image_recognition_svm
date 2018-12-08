from CVPipelines.CVPipeline import CVPipeline


class CVPipelineLinear(CVPipeline):

    def __init__(self, pipeline_type,fold1=None, fold2=None, path=None, seed=0, results=None):

        super(CVPipelineLinear, self).__init__(pipeline_type, fold1=fold1, fold2=fold2, path=path, seed=seed,
                                               results=results)
        self.kernel = "linear"

