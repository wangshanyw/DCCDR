import torch
import torch.optim as optim
from progressbar import *

import sys
import time

from OurModel import OurModel
from utils.parser import *
from utils.helpers import *
from utils.load_data import *

def train():
    print("************************* Run with following settings ***************************")
    print(args)
    print("************************************************************************************")
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    filepath = args.data_path + args.dataset + '/{}_{}'.format(args.domain_1, args.domain_2)

    n_users_1, n_items_1, user_ratings_1, n_interactions_1 = load_data(filepath=filepath+'/{}_all_item_list.dat'.format(args.domain_1))
    n_users_2, n_items_2, user_ratings_2, n_interactions_2 = load_data(filepath=filepath+'/{}_all_item_list.dat'.format(args.domain_2))
    print(n_users_1, n_items_1, n_interactions_1)
    print(n_users_2, n_items_2, n_interactions_2)

    user_ratings_test_1 = generate_test(user_ratings_1)
    user_ratings_test_2 = generate_test(user_ratings_2)


    config = dict()
    config['n_users_1'] = n_users_1
    config['n_items_1'] = n_items_1
    config['n_users_2'] = n_users_2
    config['n_items_2'] = n_items_2

    """
    *********************************************************
    Generate the Laplacian matrix, where each entry defines the decay factor (e.g., p_ui) between two connected nodes.
    """

    plain_adj_1, norm_adj_1, mean_ad_1, pre_adj_1 = get_adj_mat(filepath=filepath, dataset=args.domain_1, \
                                                                n_users=n_users_1, n_items=n_items_1, user_ratings=user_ratings_1,\
                                                                user_ratings_test=user_ratings_test_1)
    plain_adj_2, norm_adj_2, mean_adj_2, pre_adj_2 = get_adj_mat(filepath=filepath, dataset=args.domain_2, \
                                                                n_users=n_users_2, n_items=n_items_2, user_ratings=user_ratings_2,\
                                                                user_ratings_test=user_ratings_test_2)

    
    all_h_list_1, all_t_list_1, all_v_list_1 = load_adjacency_list_data(plain_adj_1)
    all_h_list_2, all_t_list_2, all_v_list_2 = load_adjacency_list_data(plain_adj_2)

    A_values_init_1 = create_initial_A_values(args.n_factors, all_v_list_1)
    A_values_init_2 = create_initial_A_values(args.n_factors, all_v_list_2)

    config['norm_adj_1'] = plain_adj_1
    config['norm_adj_2'] = plain_adj_2
    config['all_h_list_1'] = all_h_list_1
    config['all_h_list_2'] = all_h_list_2
    config['all_t_list_1'] = all_t_list_1
    config['all_t_list_2'] = all_t_list_2

    t0 = time.time()
    """
    ***********************************************************
    pretrain = 1: load embeddings with name such as embedding_xxx(.npz), l2_best_model(.npz)
    pretrain = 0: default value, no pretrained embeddings.
    """
    if args.pretrain == 1:
        print("Try to load pretrain: ", args.embed_name)
        pretrain_data = load_best(name=args.embed_name)
        if pretrain_data == None:
            print("Load pretrained model(%s)fail!" % (args.embed_name))
    else:
        pretrain_data = None

    model = OurModel(data_config=config, pretrain_data=pretrain_data, args=args, device=device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    """
    *********************************************************
    Train
    """

    best_ret_1 = np.array([0] * 6)
    best_ret_2 = np.array([0] * 6)
    
    for epoch in range(args.epoch):
        t1 = time.time()
        loss, mf_loss, emb_loss, cor_loss = 0., 0., 0., 0.
        bar_length = 300
        widgets = ['Train: ', Percentage(), ' ', Bar('#'), ' ', ETA()]
        pbar = ProgressBar(widgets=widgets, maxval=bar_length).start()
                    
        for idx in range(bar_length):
            uij_1, uij_2 = generate_train_batch_for_all_overlap(user_ratings_1, user_ratings_test_1, n_items_1, 
                                                                user_ratings_2, user_ratings_test_2, n_items_2, 
                                                                batch_size=args.batch_size)

            batch_loss = model(uij_1[:, 0], uij_1[:, 1], uij_1[:, 2], uij_2[:, 0], uij_2[:, 1], uij_2[:, 2])
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            loss += batch_loss.item() / bar_length
            pbar.update(idx)

        if np.isnan(loss) == True:
            print('ERROR: loss is nan.')
            sys.exit()

        perf_str = 'Epoch %d [%.1fs]: train==[%.5f]' % (epoch, time.time() - t1, loss)
        print(perf_str) 
        
        user_count_1 = 0
        user_count_2 = 0
        ret_1 = np.array([0.0] * 6)
        ret_2 = np.array([0.0] * 6)

        test_widgets = ['Test: ', Percentage(), ' ', Bar('#'), ' ', ETA()]
        test_pbar = ProgressBar(widgets=test_widgets, maxval=n_users_1).start()

        test_index = 0

        for t_uij_1, t_uij_2 in generate_test_batch_for_all_overlap(user_ratings_1, user_ratings_test_1, 
                                                                    n_items_1, user_ratings_2, 
                                                                    user_ratings_test_2, n_items_2):
            model.get_embeddings(t_uij_1[:, 0], t_uij_1[:, 1], t_uij_1[:, 2], t_uij_2[:, 0], t_uij_2[:, 1], t_uij_2[:, 2])
            
            u_g_embeddings_1, u_g_embeddings_2 = model.u_g_embeddings_1, model.u_g_embeddings_2
            pos_i_g_embeddings_1, pos_i_g_embeddings_2 = model.pos_i_g_embeddings_1, model.pos_i_g_embeddings_2
            neg_i_g_embeddings_1, neg_i_g_embeddings_2 = model.neg_i_g_embeddings_1, model.neg_i_g_embeddings_2

            pos_s_1 = torch.squeeze(model.rating(u_g_embeddings_1, pos_i_g_embeddings_1)).detach().cpu()
            neg_s_1 = torch.squeeze(model.rating(u_g_embeddings_1, neg_i_g_embeddings_1)).detach().cpu()
            pos_s_2 = torch.squeeze(model.rating(u_g_embeddings_2, pos_i_g_embeddings_2)).detach().cpu()
            neg_s_2 = torch.squeeze(model.rating(u_g_embeddings_2, neg_i_g_embeddings_2)).detach().cpu()
            
            user_count_1 += 1
            predictions_1 = [pos_s_1[0]]
            predictions_1 += neg_s_1
            predictions_1 = [-1 * i for i in predictions_1]

            rank_1 = np.array(predictions_1).argsort().argsort()[0]
            if rank_1 < 2:
                ret_1[0] += 1
                ret_1[3] += 1 / np.log2(rank_1 + 2)
            if rank_1 < 5:
                ret_1[1] += 1
                ret_1[4] += 1 / np.log2(rank_1 + 2)
            if rank_1 < 10:
                ret_1[2] += 1
                ret_1[5] += 1 / np.log2(rank_1 + 2)

            user_count_2 += 1
            predictions_2 = [pos_s_2[0]]
            predictions_2 += neg_s_2
            predictions_2 = [-1 * i for i in predictions_2]

            rank_2 = np.array(predictions_2).argsort().argsort()[0]
            if rank_2 < 2:
                ret_2[0] += 1
                ret_2[3] += 1 / np.log2(rank_2 + 2)
            if rank_2 < 5:
                ret_2[1] += 1
                ret_2[4] += 1 / np.log2(rank_2 + 2)
            if rank_2 < 10:
                ret_2[2] += 1
                ret_2[5] += 1 / np.log2(rank_2 + 2)
            
            test_pbar.update(test_index)
            test_index += 1
            
        best_ret_1 = best_result(best_ret_1, ret_1)
        best_ret_2 = best_result(best_ret_2, ret_2)

        print('%s: HR_2 %f HR_5 %f HR_10 %f'
                % (args.domain_1, ret_1[0]/user_count_1, ret_1[1]/user_count_1, ret_1[2]/user_count_1))
        print('%s: NDCG_2 %f NDCG_5 %f NDCG_10 %f'
                % (args.domain_1, ret_1[3] / user_count_1, ret_1[4] / user_count_1, ret_1[5] / user_count_1))
        print('Best HitRatio for %s: HR_2 %f HR_5 %f HR_10 %f'
                % (args.domain_1, best_ret_1[0]/user_count_1, best_ret_1[1]/user_count_1,
                    best_ret_1[2]/user_count_1))
        print('Best NDCG for %s: NDCG_2 %f NDCG_5 %f NDCG_10 %f'
                % (args.domain_1, best_ret_1[3]/user_count_1, best_ret_1[4]/user_count_1,
                    best_ret_1[5]/user_count_1))

        if ret_1[0] == best_ret_1[0] or ret_1[1] == best_ret_1[1] or ret_1[2] == best_ret_1[2] \
                or ret_1[3] == best_ret_1[3] or ret_1[4] == best_ret_1[4] or ret_1[5] == best_ret_1[5]:
            model_save(model.embedding_dict, args.weights_path, args, savename='best_model')
            print('save the weights in path: ', args.weights_path)

        print('%s: HR_2 %f HR_5 %f HR_10 %f'
                % (args.domain_2, ret_2[0]/user_count_2, ret_2[1]/user_count_2, ret_2[2]/user_count_2))
        print('%s: NDCG_2 %f NDCG_5 %f NDCG_10 %f'
                % (args.domain_2, ret_2[3] / user_count_2, ret_2[4] / user_count_2, ret_2[5] / user_count_2))
        print('Best HitRatio for %s: HR_2 %f HR_5 %f HR_10 %f'
                % (args.domain_2, best_ret_2[0]/user_count_2, best_ret_2[1]/user_count_2,
                    best_ret_2[2]/user_count_2))
        print('Best NDCG for %s: NDCG_2 %f NDCG_5 %f NDCG_10 %f'
                % (args.domain_2, best_ret_2[3]/user_count_2, best_ret_2[4]/user_count_2,
                    best_ret_2[5]/user_count_2))

        if ret_2[0] == best_ret_2[0] or ret_2[1] == best_ret_2[1] or ret_2[2] == best_ret_2[2] \
                or ret_2[3] == best_ret_2[3] or ret_2[4] == best_ret_2[4] or ret_2[5] == best_ret_2[5]:
            model_save(model.embedding_dict, args.weights_path, args, savename='best_model')
            print('save the weights in path: ', args.weights_path)


            
if __name__ == '__main__':
    train()
