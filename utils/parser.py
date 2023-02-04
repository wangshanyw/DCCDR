import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run MyModel.")
    parser.add_argument('--data_path', nargs='?', default='data/',
                        help='Input data path.')
    parser.add_argument('--weights_path', nargs='?', default='',
                        help='Project path.')

    parser.add_argument('--dataset', nargs='?', default='Amazon',
                        help='Choose a dataset from {Amazon, Douban}')
    parser.add_argument('--domain_1', nargs='?', default='Movie',
                        help='Choose a domain')
    parser.add_argument('--domain_2', nargs='?', default='Music',
                        help='Choose a domain')

    parser.add_argument('--pretrain', type=int, default=0,
                        help='0: No pretrain, 1:Use stored models.')
    parser.add_argument('--embed_name', nargs='?', default='best_model',
                        help='Name for pretrained model.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Interval of evaluation.')

    parser.add_argument('--alpha', type=float, default=0.001,
                        help='Number of epochs')
    parser.add_argument('--beta', type=float, default=0.001,
                        help='Number of epochs')

    parser.add_argument('--epoch', type=int, default=200,
                        help='Number of epochs')
    parser.add_argument('--embed_size', type=int, default=64,
                        help='Embedding size.')
    parser.add_argument('--layer_size', nargs='?', default='[64]',
                        help='Output sizes of every layer')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--regs', nargs='?', default='[1e-3,1e-4,1e-4]',
                        help='Regularizations.')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--multicore', type=int, default=0, help='whether we use multiprocessing or not in test')
        
    parser.add_argument('--n_layers', type=int, default=3,
                        help='Layer numbers.')
    parser.add_argument('--n_factors', type=int, default=2,
                        help='Number of factors to disentangle the original embed-size representation.')
    parser.add_argument('--n_iterations', type=int, default=2,
                        help='Number of iterations to perform the routing mechanism.')
    
    parser.add_argument('--show_step', type=int, default=2,
                        help='Test every show_step epochs.')
    parser.add_argument('--early', type=int, default=40,
                        help='Step for stopping') 

    parser.add_argument('--save_flag', type=int, default=0,
                        help='0: Disable model saver, 1: Save Better Model')
    parser.add_argument('--save_name', nargs='?', default='best_model',
                        help='Save_name.')
    
    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')

    parser.add_argument('--keep_prob', type=float, default=0.6, help='keep_prob of dropout in lightgcn')
    parser.add_argument('--A_split', type=bool, default=False, help='a_split')
    parser.add_argument('--model', nargs='?', default='dgcdr', help='rec-model')
    return parser.parse_args()


args = parse_args()