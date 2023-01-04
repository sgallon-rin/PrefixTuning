import argparse
import os

# added by sgallon
HOME_DIR = "/home/lr/shenjl/research/ref-code/PrefixTuning"
XSUM_DATA_DIR = "/home/lr/shenjl/research/ref-code/PrefixTuning/data_seq2seq"
MODEL_DIR = "/home/lr/shenjl/research/ref-code/PrefixTuning/models"
# end by sgallon

# example: python train_run.py keyword temp_keyword _
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prefix Tuning training args.')
    parser.add_argument('--old_model_name', type=str, default='facebook/bart-large',
                        help='pretrained old (frozen) model name to load from transformers',
                        choices=['facebook/bart-large', 'facebook/mbart-large-cc25'])  # added by sgallon
    parser.add_argument('--mode', type=str, default='xsum', help='',
                        choices=['xsum', 'xsum_news', 'xsum_news_sport', 'toy_xsum_10', 'japanese_xlsum'])
    parser.add_argument('--tuning_mode', type=str, default='prefixtune', help='',
                        choices=['prefixtune', 'finetune', 'finetune-top', 'bothtune', 'adaptertune'])
    parser.add_argument('--optim_prefix', type=str, default='yes', help='', choices=['yes', 'no'])
    parser.add_argument('--preseqlen', type=int, default=5, help='')
    parser.add_argument('--prefix_mode', type=str, default='activation', help='', choices=['embedding', 'activation'])
    parser.add_argument('--format_mode', type=str, default='cat', help='', choices=['cat', 'infix', 'peek', 'nopeek'])

    parser.add_argument('--dir_name', type=str, default=None, help='')
    parser.add_argument('--notes', type=str, default=None, help='')
    parser.add_argument('--lowdata_token', type=str, default='summarize', help='')
    parser.add_argument('--use_lowdata_token', type=str, default='yes', help='')

    parser.add_argument('--parametrize_emb', type=str, default='MLP', help='')
    parser.add_argument('--adapter_design', type=int, default=1, help='')
    parser.add_argument('--top_layers', type=int, default=1, help='')

    parser.add_argument('--do_train', type=str, default='yes', help='')

    parser.add_argument('--fp16', type=str, default='no', help='')

    parser.add_argument('--gpus', type=int, default=1, help='')

    # training parameters.
    parser.add_argument('--use_dropout', type=str, default='no', help='')
    parser.add_argument('--seed', type=int, default=101, help='')  # old is 42
    parser.add_argument('--bsz', type=int, default=10, help='')  # batch size
    # parser.add_argument('--use_big', type=str, default='no', help='')
    parser.add_argument('--epoch', type=int, default=5, help='')
    parser.add_argument('--max_steps', type=int, default=400, help='')
    parser.add_argument('--eval_steps', type=int, default=50, help='')
    parser.add_argument('--warmup_steps', type=int, default=100, help='')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='')
    parser.add_argument('--learning_rate', type=float, default=5e-05, help='')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='')
    parser.add_argument('--dropout', type=float, default=0.0, help='')

    parser.add_argument('--label_smoothing', type=float, default=0.0, help='')
    parser.add_argument('--length_pen', type=float, default=1.0, help='')
    parser.add_argument('--mid_dim', type=int, default=512, help='')
    parser.add_argument('--use_deep', type=str, default='no', help='')

    parser.add_argument('--prefix_model_path', type=str, default=None, help='')
    parser.add_argument('--finetune_model_path', type=str, default=None, help='')
    # parser.add_argument('--submit', type=str, default='no', help='')

    args = parser.parse_args()

    if args.optim_prefix == 'yes':
        assert args.preseqlen is not None
    if args.prefix_model_path is not None:
        load_prefix_model = True
    else:
        load_prefix_model = False

    # args.mode is dataset
    if args.mode == 'xsum':
        data_dir = os.path.join(XSUM_DATA_DIR, 'xsum')
        folder_name = os.path.join(MODEL_DIR, "xsum_models/")
        max_source_length = 1024
        max_target_length = 60
        val_max_target_length = 60
        test_max_target_length = 100
        xsum_app = ' --max_source_length {} --max_target_length {} --val_max_target_length {} ' \
                   '--test_max_target_length {} '.format(max_source_length, max_target_length,
                                                         val_max_target_length, test_max_target_length)
        if args.fp16 == 'yes':
            xsum_app += ' --fp16 --fp16_opt_level O1 '
        assert args.optim_prefix == 'yes'
    elif args.mode == 'xsum_news':
        data_dir = os.path.join(XSUM_DATA_DIR, 'xsum_news')
        folder_name = os.path.join(MODEL_DIR, "xsum_news_models/")
        max_source_length = 512
        max_target_length = 60
        val_max_target_length = 60
        test_max_target_length = 100
        xsum_app = ' --max_source_length {} --max_target_length {} --val_max_target_length {} ' \
                   '--test_max_target_length {} '.format(max_source_length, max_target_length,
                                                         val_max_target_length, test_max_target_length)
        if args.fp16 == 'yes':
            xsum_app += ' --fp16 --fp16_opt_level O1 '
    elif args.mode == 'xsum_news_sport':
        data_dir = os.path.join(XSUM_DATA_DIR, 'xsum_topic-news-sports')
        folder_name = os.path.join(MODEL_DIR, "xsum_news_sport_models/")
        max_source_length = 512
        max_target_length = 60
        val_max_target_length = 60
        test_max_target_length = 100
        xsum_app = ' --max_source_length {} --max_target_length {} --val_max_target_length {} ' \
                   '--test_max_target_length {} '.format(max_source_length, max_target_length,
                                                         val_max_target_length, test_max_target_length)
        if args.fp16 == 'yes':
            xsum_app += ' --fp16 --fp16_opt_level O1 '
    elif args.mode == 'toy_xsum_10':  # toy dataset for test by sgallon
        data_dir = os.path.join(XSUM_DATA_DIR, 'toy_xsum_10')
        folder_name = os.path.join(MODEL_DIR, "toy_xsum_10/")
        max_source_length = 1024
        max_target_length = 60
        val_max_target_length = 60
        test_max_target_length = 100
        xsum_app = ' --max_source_length {} --max_target_length {} --val_max_target_length {} ' \
                   '--test_max_target_length {} '.format(max_source_length, max_target_length,
                                                         val_max_target_length, test_max_target_length)
        if args.fp16 == 'yes':
            xsum_app += ' --fp16 --fp16_opt_level O1 '
    elif args.mode == 'japanese_xlsum':  # japanese xlsum dataset
        data_dir = os.path.join(XSUM_DATA_DIR, 'japanese_xlsum')
        folder_name = os.path.join(MODEL_DIR, "japanese_xlsum/")
        max_source_length = 1024
        max_target_length = 60
        val_max_target_length = 60
        test_max_target_length = 100
        xsum_app = ' --max_source_length {} --max_target_length {} --val_max_target_length {} ' \
                   '--test_max_target_length {} '.format(max_source_length, max_target_length,
                                                         val_max_target_length, test_max_target_length)
        if args.fp16 == 'yes':
            xsum_app += ' --fp16 --fp16_opt_level O1 '
        xsum_app += ' --src_lang ja_XX --tgt_lang ja_XX '  # japanese lang for mBART
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)

    batch_size = args.gradient_accumulation_steps * args.bsz

    if args.dir_name is None:
        Model_FILE = args.mode + args.tuning_mode + '_' + args.optim_prefix[:1] + '_' + str(args.preseqlen) + \
                     '_' + args.prefix_mode[:3] + '_' + args.format_mode[:3] + '_' + \
                     'b={}-'.format(batch_size) + 'e={}_'.format(args.epoch) + 'd={}_'.format(args.dropout) + \
                     'l={}_'.format(args.label_smoothing) + 'lr={}_'.format(args.learning_rate) \
                     + 'w={}_'.format(args.weight_decay) + 's={}'.format(args.seed) + '_d={}'.format(
            args.use_deep[:1]) + \
                     '_m={}'.format(args.mid_dim)
    else:
        Model_FILE = args.dir_name

    if args.notes is not None:
        Model_FILE += '_{}'.format(args.notes)

    logging_dir = os.path.join(folder_name, 'runs', Model_FILE)
    Model_FILE = '{}{}'.format(folder_name, Model_FILE)
    # print(Model_FILE)

    # OLD_MODEL = 'facebook/bart-large'
    OLD_MODEL = args.old_model_name

    app = "--optim_prefix {} --preseqlen {} --prefix_mode {} --format_mode {} " \
          "--gradient_accumulation_steps {} --learning_rate {} --weight_decay {} --seed {} " \
          "--mid_dim {} --use_dropout {} --prefix_dropout {} ". \
        format(args.optim_prefix, args.preseqlen, args.prefix_mode, args.format_mode,
               args.gradient_accumulation_steps, args.learning_rate, args.weight_decay, args.seed,
               args.mid_dim, args.use_dropout, args.dropout)

    if args.prefix_mode == 'embedding':
        app += ' --parametrize_emb {} '.format(args.parametrize_emb)

    if args.tuning_mode == 'adaptertune':
        app += ' --adapter_design {} '.format(args.adapter_design)

    app += xsum_app

    # if OLD_MODEL == 'gpt2-large':
    #     app += ' --cache_dir /u/scr/xlisali/contrast_LM/transformers/examples/control/gpt2-large-s3 '

    if args.tuning_mode == 'finetune-top':
        app += ' --top_layers {} '.format(args.top_layers)

    controlprefix = ('yes' if args.tuning_mode == 'prefixtune' else 'no')

    if args.do_train == 'yes':
        COMMANDLINE = 'python finetune.py ' \
                      '--model_name_or_path {} ' \
                      '--output_dir {} ' \
                      '--data_dir {} ' \
                      '--tuning_mode {} ' \
                      '--preseqlen {} ' \
                      '--do_train ' \
                      '--label_smoothing {} ' \
                      '--use_deep {} ' \
                      '--gpus {} ' \
                      '--learning_rate {} ' \
                      '--train_batch_size {} ' \
                      '--eval_batch_size {} ' \
                      '--num_train_epochs {} '.format(OLD_MODEL, Model_FILE, data_dir, args.tuning_mode, args.preseqlen,
                                                      args.label_smoothing, args.use_deep,
                                                      args.gpus, args.learning_rate, args.bsz, args.bsz, args.epoch)
    else:
        if args.tuning_mode == 'finetune':
            assert args.finetune_model_path is not None
            print('loading from the finetune model {}'.format(args.finetune_model_path))
            Model_FILE = args.finetune_model_path + '_decode_eval' + '_{}'.format(args.length_pen)
            print('writing the decoded results to {}'.format(Model_FILE))
            COMMANDLINE = 'python finetune.py ' \
                          '--model_name_or_path {} ' \
                          '--output_dir {} ' \
                          '--data_dir {} ' \
                          '--tuning_mode {} ' \
                          '--preseqlen {} ' \
                          '--do_predict ' \
                          '--use_deep {} ' \
                          '--gpus {} ' \
                          '--train_batch_size {} ' \
                          '--eval_batch_size {} ' \
                          '--length_penalty {} ' \
                          '--num_train_epochs {} '.format(args.finetune_model_path, Model_FILE, data_dir,
                                                          args.tuning_mode, args.preseqlen, args.use_deep, args.gpus,
                                                          10, 10, args.length_pen, args.epoch)
        else:
            assert args.prefix_model_path is not None
            print('loading from the prefix model {}'.format(args.prefix_model_path))
            print('loading from the main model {}'.format(OLD_MODEL))
            Model_FILE = args.prefix_model_path + '_decode_eval' + '_{}'.format(args.length_pen)
            print('writing the decoded results to {}'.format(Model_FILE))
            COMMANDLINE = 'python finetune.py ' \
                          '--model_name_or_path {} ' \
                          '--prefixModel_name_or_path {} ' \
                          '--output_dir {} ' \
                          '--data_dir {} ' \
                          '--tuning_mode {} ' \
                          '--preseqlen {} ' \
                          '--do_predict ' \
                          '--use_deep {} ' \
                          '--gpus {} ' \
                          '--train_batch_size {} ' \
                          '--eval_batch_size {} ' \
                          '--seed {} ' \
                          '--length_penalty {} ' \
                          '--num_train_epochs {} '.format(OLD_MODEL, args.prefix_model_path, Model_FILE, data_dir,
                                                          args.tuning_mode, args.preseqlen, args.use_deep, args.gpus,
                                                          8, 8, args.seed, args.length_pen, args.epoch)

    COMMANDLINE += app

    with open(Model_FILE + '.sh', 'w') as f:
        print(COMMANDLINE, file=f)

    print(COMMANDLINE)
    os.system(COMMANDLINE)

