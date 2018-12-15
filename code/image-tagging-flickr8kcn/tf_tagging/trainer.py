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


from constant import *
import utility
from dataset import DataBatchIterator
from mlp import MLP


logger = logging.getLogger(__file__)
formatter_log = "[%(asctime)s - %(filename)s:line %(lineno)s] %(message)s"
logging.basicConfig(
    format = formatter_log,
    datefmt = '%d %b %H:%M:%S')
logger.setLevel(logging.INFO)


flags = tf.app.flags

# GPU option
flags.DEFINE_integer("ses_threads", 4, "Tensorflow CPU session threads to use")
flags.DEFINE_float("gpu_memory_fraction", 0.05, "Fraction of GPU memory to use")
flags.DEFINE_integer("gpu", 0, "select a GPU, 0:gpu0 1:gpu1 (default: 0)")

flags.DEFINE_string("rootpath", ROOT_PATH, "rootdir of the data and model (default: %s)"%ROOT_PATH)
flags.DEFINE_string('train_collection', 'msrvtt10ktrain', 'collection dataset for model training')
flags.DEFINE_string('val_collection', 'msrvtt10kval', 'collection dataset for model validation')
flags.DEFINE_string("model_name", DEFAULT_MODEL_NAME, "model configuration (default: %s)" % DEFAULT_MODEL_NAME)
flags.DEFINE_string('annotation_name', 'conceptsbigram288.txt', 'annotation / lablel name list')
flags.DEFINE_string('vf_name', 'mean_pyresnet-101_rbps13k_step_3_aug_1_dist_2x2,resnet-imagenet-101-0,39,flatten0_output', 'name of the visual feature')
flags.DEFINE_integer('overwrite', 0, 'overwrite existing file (default: 0)')
flags.DEFINE_integer('multi_task', 0, 'multi task learning (default: 0)')
flags.DEFINE_string('aux_train_collection', '', 'auxiliary collection dataset for model training')
flags.DEFINE_string('aux_annotation_name', '', 'auxiliary annotation / lablel name list')

FLAGS = flags.FLAGS


def run_epoch(session, model, train_dataset, val_dataset, multi_task=False):
    train_loss_list = [0] * train_dataset.num_batches
    progbar = utils.Progbar(train_dataset.num_batches, verbose=1)
    
    for minibatch_index, (ids_train, X_train, Y_train, YE_train) in enumerate(train_dataset):
        if multi_task:
            loss, _ = session.run([model.loss, model.train_op], 
                feed_dict={model.input_data: X_train, model.targets: Y_train, model.aux_targets:YE_train}) 
        else:
            loss, _ = session.run([model.loss, model.train_op], 
                feed_dict={model.input_data: X_train, model.targets: Y_train})
        progbar.add(1, values=[("train loss", loss)])
        train_loss_list[minibatch_index] = loss

    overall_perf = np.zeros((3,4))
    progbar = utils.Progbar(val_dataset.num_batches)
    n = 0             
    for minibatch_index, (ids_val, X_val, Y_val, _) in enumerate(val_dataset):
        valid_pred_top_indices = session.run(model.pred_top_indices, 
            feed_dict={model.input_data: X_val, model.targets: Y_val})
        res = utility.compute_hit_precision_recall_f1(valid_pred_top_indices, Y_val)
        overall_perf += res * X_val.shape[0]
        n += X_val.shape[0]
        perf_so_far = overall_perf / n # hit5, p5, r5 f5
        progbar.add(1, values=zip(str.split("hit5 p5 recall5"), perf_so_far[1,:3]))

    train_loss = np.mean(train_loss_list)
    val_perf = utility.convert_to_one_metric(perf_so_far)

    return (train_loss,  val_perf) 


def main(unused_args):
    model_dir = utility.get_model_dir(FLAGS)
    if os.path.exists(model_dir) and not FLAGS.overwrite:
        logger.info('%s exists. quit', model_dir)
        sys.exit(0)

    rootpath = FLAGS.rootpath
    overwrite = FLAGS.overwrite
    train_collection = FLAGS.train_collection
    aux_train_collection = FLAGS.aux_train_collection
    val_collection = FLAGS.val_collection
    vf_name = FLAGS.vf_name
    annotation_name = FLAGS.annotation_name
    aux_annotation_name = FLAGS.aux_annotation_name
    # Load model configuration
    config_path = os.path.join(os.path.dirname(__file__), 'model_conf', FLAGS.model_name + '.py')
    config = utility.load_config(config_path)
    
    train_loss_hist_file = os.path.join(model_dir, 'train_loss_hist.txt')
    val_per_hist_file = os.path.join(model_dir, '%s.txt' % val_collection)

    if os.path.exists(train_loss_hist_file) and not overwrite:
        logger.info('%s exists. quit', train_loss_hist_file)
        sys.exit(0)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    else:
        os.system('rm -rf %s/*' % model_dir)

    collections = [train_collection, aux_train_collection] if FLAGS.multi_task else [train_collection]
    annotation_names = [annotation_name, aux_annotation_name] if FLAGS.multi_task else [annotation_name]
    concept_files = [os.path.join(rootpath, col, 'Annotations', anno) for (col,anno) in zip(collections, annotation_names)]
    train_dataset = DataBatchIterator(collections, concept_files, vf_name, config.batch_size, rootpath)
    val_dataset = DataBatchIterator([val_collection], concept_files[:1], vf_name, config.batch_size, rootpath)

    aux_classes = train_dataset.aux_num_labels


    # Start model training
    gpu_options = tf.GPUOptions(
          per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction)
    config_proto = tf.ConfigProto(
        intra_op_parallelism_threads=FLAGS.ses_threads, gpu_options=gpu_options, allow_soft_placement=True)
    
    with tf.Graph().as_default(), tf.Session(config=config_proto) as session:
      with tf.variable_scope("MLPModel"):
        model = MLP(is_training=True,
                     n_layers=[train_dataset.feat_dim, config.hidden_size, train_dataset.num_labels],
                     last_nl_fun=config.output,
                     loss_fun=config.loss,
                     keep_prob=config.keep_prob,
                     top_k=config.top_k,
                     aux_classes=aux_classes,
                     flag_with_saver=True,
                     gpu=FLAGS.gpu)
      
      if tf.__version__ < '1.0':
          tf.initialize_all_variables().run()
      else:
          tf.global_variables_initializer().run()
      model.assign_lr(session, config.learning_rate)


      best_val_perf = 0
      count_down = config.tolerate
      train_loss_hist = []
      val_per_hist = []
      learning_rate = config.learning_rate

      for epoch in range(config.max_epochs):
          logger.info('epoch %d', epoch)
          train_dataset.shuffle()
          train_loss, val_perf = run_epoch(session, model, train_dataset, val_dataset, FLAGS.multi_task)
          logger.info('train loss %.4f, val perf %.4f', train_loss, val_perf)
          train_loss_hist.append(train_loss)
          val_per_hist.append(val_perf)
 
          if val_perf > best_val_perf:
              best_val_perf = val_perf
              best_ckpt_name = 'epoch_%d.ckpt' % epoch
              model_file = os.path.join(model_dir, best_ckpt_name)
              model.saver.save(session, model_file)
              logger.info('*** better model -> val_perf=%.4f', val_perf)
          else:
              count_down -= 1
              if 0 == count_down:
                  if learning_rate > 1e-8:
                      count_down = config.tolerate
                      learning_rate /= 2.0
                      model.assign_lr(session, learning_rate)
                      logger.info('adjust lr to %g', learning_rate)
                  else:
                      logger.info("early-stop happend at epoch %d", epoch)
                      break

      with open(train_loss_hist_file, 'w') as fout:
          for i, loss in enumerate(train_loss_hist):
              fout.write("epoch_" + str(i) + " " + str(loss) + "\n")
          fout.close()

      sorted_epoch_perf = sorted(zip(range(len(val_per_hist)), val_per_hist), key = lambda x: x[1], reverse=True)
      with open(val_per_hist_file, 'w') as fout:
          for i, perf in sorted_epoch_perf:
              fout.write("epoch_" + str(i) + " " + str(perf) + "\n")
          fout.close()



if __name__ == '__main__':
    tf.app.run()
