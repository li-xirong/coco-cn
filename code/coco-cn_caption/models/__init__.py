from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .FCModel import FCModel

def setup(opt):
    # FC model
    if opt.model_name == 'fc':
        model = FCModel(opt)
    else:
        raise Exception(
            "Caption model not supported: {}".format(opt.model_name))

    return model
