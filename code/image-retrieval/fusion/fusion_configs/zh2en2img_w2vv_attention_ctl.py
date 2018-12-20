
class config(object):
    rank_norm = 0
    run_list = str.split("zh2en_w2vv_attention_ctl w2vv_attention_ctl")
    nr_of_runs = len(run_list)
    #weights = [1.0/nr_of_runs] * nr_of_runs
    weights = [0.66, 0.34]


