from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import random
import numpy as np
import logging
import tensorflow as tf
from tensorflow.contrib.keras import utils 


from bigfile import BigFile
from constant import *
import utility
from dataset import TestDataBatchIterator
from mlp import MLPPredictor

logger = logging.getLogger(__file__)
formatter_log = "[%(asctime)s - %(filename)s:line %(lineno)s] %(message)s"
logging.basicConfig(
    format = formatter_log,
    datefmt = '%d %b %H:%M:%S')
logger.setLevel(logging.INFO)


flags = tf.app.flags

# GPU option
flags.DEFINE_integer("ses_threads", 2, "Tensorflow CPU session threads to use")
flags.DEFINE_float("gpu_memory_fraction", 0.05, "Fraction of GPU memory to use")
flags.DEFINE_integer("gpu", 1, "select a GPU, 0:gpu0 1:gpu1 (default: 1)")

flags.DEFINE_string("rootpath", ROOT_PATH, "rootdir of the data and model (default: %s)"%ROOT_PATH)
flags.DEFINE_string('train_collection', 'msrvtt10ktrain', 'collection dataset for model training')
flags.DEFINE_string('val_collection', 'msrvtt10kval', 'collection dataset for model validation')
flags.DEFINE_string('test_collection', 'msrvtt10ktest', 'collection dataset for evaluating the performance')
flags.DEFINE_string("model_name", DEFAULT_MODEL_NAME, "model configuration (default: %s)" % DEFAULT_MODEL_NAME)
flags.DEFINE_string('annotation_name', 'conceptsbigram288.txt', 'annotation / lablel name list')
flags.DEFINE_string('vf_name', 'mean_pyresnet-101_rbps13k_step_3_aug_1_dist_2x2,resnet-imagenet-101-0,39,flatten0_output', 'name of the visual feature')
flags.DEFINE_integer('overwrite', 0, 'overwrite existing file (default: 0)')
flags.DEFINE_integer('multi_task', 0, 'multi task learning (default: 0)')

flags.DEFINE_integer('batch_size', 256, 'size of test batch')
flags.DEFINE_integer('top_k', 10, 'the number of top predicted labels')
flags.DEFINE_string('eval_k', '1-5-10', 'the list of top k for evaluation')

flags.DEFINE_string('aux_train_collection', '', 'auxiliary collection dataset for model training')
flags.DEFINE_string('aux_annotation_name', '', 'auxiliary annotation / lablel name list')

FLAGS = flags.FLAGS


def test():
    rootpath = FLAGS.rootpath
    overwrite = FLAGS.overwrite
    train_collection = FLAGS.train_collection
    aux_train_collection = FLAGS.aux_train_collection
    val_collection = FLAGS.val_collection
    test_collection = FLAGS.test_collection
    vf_name = FLAGS.vf_name
    annotation_name = FLAGS.annotation_name
    aux_annotation_name = FLAGS.aux_annotation_name
    top_k = FLAGS.top_k
    
    config_path = os.path.join(os.path.dirname(__file__), 'model_conf', FLAGS.model_name + '.py')
    config = utility.load_config(config_path)
    
    model_dir = utility.get_model_dir(FLAGS)
    val_perf_file = os.path.join(model_dir, '%s.txt' % val_collection)
    epoch, val_perf = open(val_perf_file).readline().strip().split()
    epoch = int(epoch.rsplit('_', 1)[1])
    model_path = os.path.join(model_dir, 'epoch_%d.ckpt' % epoch)

    test_data_iter = TestDataBatchIterator(test_collection, vf_name, FLAGS.batch_size, rootpath=rootpath)

    res_dir = utility.get_pred_dir(FLAGS)
    res_file = os.path.join(res_dir, 'id.tagvotes.txt')

    if os.path.exists(res_file) and not overwrite:
        logger.info('%s exists. quit', res_file)
        return 

    concept_file = os.path.join(rootpath, train_collection, 'Annotations', annotation_name)
    concepts = map(str.strip, open(concept_file).readlines())
    
    if FLAGS.multi_task:
        aux_concept_file = os.path.join(rootpath, aux_train_collection, 'Annotations', aux_annotation_name)
        aux_concepts = map(str.strip, open(aux_concept_file).readlines())
    else:
        aux_concepts = []
    
    mlp_predictor = MLPPredictor(n_layers=[test_data_iter.feat_dim, config.hidden_size, len(concepts)],
                                  last_nl_fun=config.output,
                                  loss_fun=config.loss,
                                  top_k=min(len(concepts),FLAGS.top_k),
                                  ses_threads=FLAGS.ses_threads,
                                  aux_classes=len(aux_concepts),
                                  gpu_memory_fraction=FLAGS.gpu_memory_fraction,
                                  gpu=FLAGS.gpu)
      
    mlp_predictor.load_model(model_path)


    progbar = utils.Progbar(test_data_iter.num_batches)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    fout = open(res_file, "w")
    
    for minibatch_index, (ids_test, X_test, _, _) in enumerate(test_data_iter):
        pred_top_values_batch, pred_top_indices_batch = mlp_predictor.predict(X_test)

        for idx, img_id in enumerate(ids_test):
            tagvotes = zip(pred_top_indices_batch[idx], pred_top_values_batch[idx])
            res = ' '.join(['%s %g' % (concepts[x[0]], x[1]) for x in tagvotes])
            fout.write('%s %s\n' % (img_id, res))
        progbar.add(1)
    fout.close()


def main(unused_args):
    test()


if __name__ == '__main__':
    tf.app.run()
