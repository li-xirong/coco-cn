import argparse
import json

def parse_opt():
    parser = argparse.ArgumentParser()
    # Data input settings

    parser.add_argument('--rootpath', type=str, default='../data',
                    help='rootpath of the data and models')

    parser.add_argument('--collection', type=str, default='flickr8kboson',
                    help='collection name (e.g.: flickr8kboson)')
    parser.add_argument('--vf_name', type=str, default='pyresnet152-pool5osl2',
                    help='visual feature name')

    parser.add_argument('--start_from', type=str, default=None,
                    help='continue training from saved model at this path.')
    parser.add_argument('--seq_length', type=int, default=20,
                    help='max sequence length of the caption')
    parser.add_argument('--shuffle', type=int, default=1,
                    help='shuffle the data? (1=yes,0=no, default=1)')
    parser.add_argument('--num_workers', type=int, default=10,
                    help='use how many workers to load the data?')

    # Model settings
    parser.add_argument('--model_name', type=str, default="fc",
                    help='fc, topdown')
    parser.add_argument('--use_att', type=int, default=0,
                    help='use attention machanism? (1=yes,0=no, default=0)')
    parser.add_argument('--cross_lingual_similarity', type=float, default=0.0,
                    help='weight of cross lingual similarity as reward (default=0.0)')


    parser.add_argument('--lstm_hidden_size', type=int, default=512,
                    help='size of the lstm in number of hidden nodes in each layer')
    parser.add_argument('--lstm_num_layers', type=int, default=1,
                    help='number of layers in the lstm')
    parser.add_argument('--lstm_input_size', type=int, default=512,
                    help='lstm inpt size')
    parser.add_argument('--word_embed_size', type=int, default=512,
                    help='the encoding size of each token in the vocabulary, and the image.')

    parser.add_argument('--fc_feat_size', type=int, default=2048,
                    help='2048 for resnet, 4096 for vgg')
    parser.add_argument('--att_feat_size', type=int, default=2048,
                    help='2048 for resnet, 512 for vgg')
    parser.add_argument('--att_hid_size', type=int, default=512,
                    help='the hidden size of the attention MLP; only useful in show_attend_tell; 0 if not using hidden layer')

    parser.add_argument('--bilingual', type=int, default=0,
                    help='whether the eng and chn sentences are concatenated (default=0)')


    # Optimization: General
    parser.add_argument('--num_epochs', type=int, default=50,
                    help='max number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                    help='minibatch size')
    parser.add_argument('--grad_clip', type=float, default=0.1, #5.,
                    help='clip gradients at this value')
    parser.add_argument('--lstm_drop_prob', type=float, default=0.5,
                    help='strength of dropout in the Language Model RNN')

    parser.add_argument('--beam_size', type=int, default=1,
                    help='used when sample_max = 1, indicates number of beams in beam search. Usually 2 or 3 works well. More is not better. Set this to 1 for faster runtime but a bit worse performance.')

    #Optimization: for the Language Model
    parser.add_argument('--learning_rate', type=float, default=4e-4,
                    help='learning rate')
    parser.add_argument('--learning_rate_decay_start', type=int, default=0, 
                    help='at what iteration to start decaying learning rate? (-1 = dont) (in epoch)')
    parser.add_argument('--learning_rate_decay_every', type=int, default=3, 
                    help='every how many iterations thereafter to drop LR?(in epoch)')
    parser.add_argument('--learning_rate_decay_rate', type=float, default=0.8, 
                    help='every how many iterations thereafter to drop LR?(in epoch)')

    parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight_decay')

    parser.add_argument('--feedback_start_epoch', type=int, default=-1, 
                    help='at what iteration to start decay gt probability')
    parser.add_argument('--feedback_prob_increase_every', type=int, default=5, 
                    help='every how many iterations thereafter to gt probability')
    parser.add_argument('--feedback_prob_start', type=float, default=0.0,     
                    help='From how much to start the feedback prob')
    parser.add_argument('--feedback_prob_increment', type=float, default=0.05, 
                    help='How much to update the prob')
    parser.add_argument('--feedback_prob_max', type=float, default=0.25, 
                    help='Maximum scheduled sampling prob.')

    parser.add_argument('--language_eval', type=int, default=0,
                    help='Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
    parser.add_argument('--use_merged_vocab', type=int, default=0,
                    help='whether to use merged vocab file for sequential learning (1 = yes, 0 = no)?')

    '''
    # Evaluation/Checkpointing
    #parser.add_argument('--train_collection', type=str, default=None,
    #                help='Train collection for getting the model path to be tested; if None, it means its the same with collection')
    parser.add_argument('--val_images_use', type=int, default=-1,
                    help='how many images to use when periodically evaluating the validation loss? (-1 = all)')
    parser.add_argument('--save_checkpoint_every', type=int, default=2500,
                    help='how often to save a model checkpoint (in iterations)?')
    # parser.add_argument('--checkpoint_path', type=str, default='save',
    #                 help='directory to store checkpointed models')
    parser.add_argument('--losses_log_every', type=int, default=25,
                    help='How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')       
    parser.add_argument('--load_best_score', type=int, default=1,
                    help='Do we load previous best score when resuming training.')       
    parser.add_argument('--dump_json', type=int, default=1,
                help='Dump json with predictions into vis folder? (1=yes,0=no)')
    parser.add_argument('--dump_path', type=int, default=0,
                help='Write image paths along with predictions into vis json? (1=yes,0=no)')
    parser.add_argument('--sample_max', type=int, default=1,
                help='1 = sample argmax words. 0 = sample from distributions.')
    parser.add_argument('--temperature', type=float, default=1.0,
                help='temperature when sampling from distributions (i.e. when sample_max = 0). Lower = "safer" predictions.')
    '''
    # misc
    parser.add_argument('--id', type=str, default='',
                    help='an id identifying this run/job. used in cross-val and appended when writing progress files')

    args = parser.parse_args()

    if args.model_name == 'fc':
        args.use_att = False
    else:
        args.use_att = True


    # print args
    print json.dumps(vars(args), indent = 2)
    # Check if args are valid
    
    assert args.lstm_hidden_size > 0, "lstm_hidden_size should be greater than 0"
    assert args.lstm_num_layers > 0, "num_layers should be greater than 0"
    assert args.batch_size > 0, "batch_size should be greater than 0"
    assert args.lstm_drop_prob >= 0 and args.lstm_drop_prob < 1, "lstm_drop_prob should be between 0 and 1"
    assert args.beam_size > 0, "beam_size should be greater than 0"
    assert args.language_eval == 0 or args.language_eval == 1, "language_eval should be 0 or 1"

    
    return args
