import os
import itertools
import numpy as np
from sentence_transformers import SentenceTransformer, util

MODEL_NAMES = {
    "mpnet": "../../tmp/all-mpnet-base-v2"
}
 
class Retriever(object):
    def __init__(self, args):
        super(Retriever, self).__init__()

        self.args = args

        self.set_cache_dir(self.args.index_dir)
        self.model = self.get_model()

    def set_cache_dir(self, dir):
        os.environ['PYSERINI_CACHE'] = dir

    def get_model(self):
        return None
    
    def retrieve(self, samples, return_ids=True, offset=0, *args, **kwargs):
        results = [
            self.model.search(sample['question'])[offset:]
            for sample in samples
        ]

        raw_results = [
            [(doc.docid, doc.score) for doc in docs] 
            for docs in results
        ]

        ver_results = [
            [doc.raw for doc in docs]
            for docs in results
        ]

        return ver_results if return_ids == False \
               else (ver_results, raw_results)