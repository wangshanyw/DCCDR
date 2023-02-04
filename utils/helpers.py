import torch
import numpy as np


def load_best(args, name="best_model"):
    pretrain_path='{}pretrain/{}/{}.npz'.format(args.proj_path, args.dataset+'_'+args.domain_1+'_'+args.domain_2, name)
    try:
        pretrain_data = torch.load(pretrain_path)
        print('load the best model: ', name)
    except Exception:
        pretrain_data = None
    return pretrain_data

def model_save(weights, path, args, savename='best_model'):
    save_pretrain_path = '%spretrain/%s/%s' % (path, args.dataset+'_'+args.domain_1+'_'+args.domain_2, savename)
    np.savez(save_pretrain_path, user_embed_1=weights['user_embedding_1'].detach().cpu().numpy(),
                                 item_embed_1=weights['item_embedding_1'].detach().cpu().numpy(),
                                 user_embed_2=weights['user_embedding_2'].detach().cpu().numpy(),
                                 item_embed_2=weights['item_embedding_2'].detach().cpu().numpy())

def best_result(best, current):
    # print("find the best number:")
    num_ret = len(best)
    ret_best = [0.0]*num_ret
    for numIdx in range(num_ret):
        ret_best[numIdx] = max(float(current[numIdx]), float(best[numIdx]))
    return ret_best