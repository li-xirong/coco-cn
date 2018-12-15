import os
import numpy as np

def load_config(config_path):
    variables = {}
    exec(compile(open(config_path, "rb").read(), config_path, 'exec'), variables)
    return variables['config']


def get_concept_file(collection, annotation_name, rootpath):
    return os.path.join(rootpath, collection, 'Annotations', annotation_name)

def get_feat_dir(collection, feature, rootpath):
    return os.path.join(rootpath, collection, 'FeatureData', feature)

def get_train_feat_dir(FLAGS):
    return get_feat_dir(FLAGS.train_collection, FLAGS.vf_name, FLAGS.rootpath)

def get_val_feat_dir(FLAGS):
    return get_feat_dir(FLAGS.val_collection, FLAGS.vf_name, FLAGS.rootpath)

def get_test_feat_dir(FLAGS):
    return get_feat_dir(FLAGS.test_collection, FLAGS.vf_name, FLAGS.rootpath)

def get_model_dir(FLAGS):
    if FLAGS.multi_task:
        return os.path.join(FLAGS.rootpath, FLAGS.train_collection, 'Models', FLAGS.aux_train_collection, 
                FLAGS.val_collection, FLAGS.annotation_name, FLAGS.aux_annotation_name, FLAGS.model_name, FLAGS.vf_name)
    else:
        return os.path.join(FLAGS.rootpath, FLAGS.train_collection, 'Models', FLAGS.val_collection, 
                FLAGS.annotation_name, FLAGS.model_name, FLAGS.vf_name)


def get_pred_dir(FLAGS):
    if FLAGS.multi_task:
        return os.path.join(FLAGS.rootpath, FLAGS.test_collection, 'autotagging', FLAGS.test_collection, FLAGS.annotation_name, FLAGS.aux_annotation_name,
                FLAGS.train_collection, FLAGS.aux_train_collection, FLAGS.val_collection, FLAGS.model_name, FLAGS.vf_name)
    else:
        return os.path.join(FLAGS.rootpath, FLAGS.test_collection, 'autotagging', FLAGS.test_collection, FLAGS.annotation_name,
                FLAGS.train_collection, FLAGS.val_collection, FLAGS.model_name, FLAGS.vf_name)



'''
perf_table = 
    hit1  p1  recall1 f1
    hit5  p5  recall5 f5
    hit10 p10 recall10 f10
'''
def convert_to_one_metric(perf_table):
    #hit, precision, recall = perf_so_far.mean(axis=0)
    hit5, p5, r5, f5 = perf_table[1,:]
    #f = 2*p5*r5/(p5+r5+1e-10)
    return p5

'''
return 
    hit1  p1  recall1 f1
    hit5  p5  recall5 f5
    hit10 p10 recall10 f10
'''
def compute_hit_precision_recall_f1(pred_labels, y_true, ranks=[1, 5, 10]):
    n_samples = y_true.shape[0]
    relevant = y_true.sum(axis=1) + 1e-10
    res = []

    for r in ranks:
        matched = np.zeros(n_samples, dtype=int)
        for i in range(n_samples):
            matched[i] = np.sum(y_true[i, pred_labels[i,:r]])
        
        hit = np.mean([x>0 for x in matched])
        _prec = matched / float(r)
        precision = np.mean(_prec)
        _rec = np.divide(matched, relevant)
        recall = np.mean(_rec)
        f_measure = np.mean( 2*_prec*_rec / (_prec+_rec+1e-10))
        res.append((hit, precision, recall, f_measure))
    return np.asarray(res)


if __name__ == '__main__':
    config_path = 'model_conf/baseline.py'
    print load_config(config_path).keep_prob

