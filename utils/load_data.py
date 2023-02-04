from collections import defaultdict
import numpy as np
import random as rd
import scipy.sparse as sp

def load_data(filepath):
    n_users = 0
    n_items = 0
    n_interactions = 0
    with open(filepath) as f:
        for l in f.readlines():
            if len(l) > 0:
                n_users += 1
                l = l.strip('\n').rstrip(' ')
                items = [int(i) for i in l.split(' ')]
                n_items = max(n_items, max(items))
                n_interactions += len(items)
    n_items += 1
    user_ratings = defaultdict(dict)
    with open(filepath) as f:
        for l in f.readlines():
            if len(l) == 0:
                break
            l = l.strip('\n').rstrip(' ')
            items = [int(i) for i in l.split(' ')]
            uid, train_items = items[0], items[1:]
            user_ratings[uid] = train_items
    return n_users, n_items, user_ratings, n_interactions


def generate_test(all_user_ratings):
    ratings_test = {}
    for user in all_user_ratings:
        ratings_test[user] = rd.sample(all_user_ratings[user], 1)[0]
    return ratings_test


def get_adj_mat(filepath, dataset, n_users, n_items, user_ratings, user_ratings_test):
    try:
        adj_mat = sp.load_npz(filepath + '/{}_adj_mat.npz'.format(dataset))
        norm_adj_mat = sp.load_npz(filepath + '/{}_norm_adj_mat.npz'.format(dataset))
        mean_adj_mat = sp.load_npz(filepath + '/{}_mean_adj_mat.npz'.format(dataset))
        print('already load adj matrix', adj_mat.shape)

    except Exception:
        adj_mat, norm_adj_mat, mean_adj_mat = create_adj_mat(n_users, n_items, user_ratings, user_ratings_test)
        sp.save_npz(filepath + '/{}_adj_mat.npz'.format(dataset), adj_mat)
        sp.save_npz(filepath + '/{}_norm_adj_mat.npz'.format(dataset), norm_adj_mat)
        sp.save_npz(filepath + '/{}_mean_adj_mat.npz'.format(dataset), mean_adj_mat)

    try:
        pre_adj_mat = sp.load_npz(filepath + '/{}_pre_adj_mat.npz'.format(dataset))
    except Exception:
        adj_mat=adj_mat
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)

        norm_adj = d_mat_inv.dot(adj_mat)
        norm_adj = norm_adj.dot(d_mat_inv)
        print('generate pre adjacency matrix.')
        pre_adj_mat = norm_adj.tocsr()
        sp.save_npz(filepath + '/{}_pre_adj_mat.npz'.format(dataset), norm_adj)

    return adj_mat, norm_adj_mat, mean_adj_mat, pre_adj_mat


def create_adj_mat(n_users, n_items, user_ratings, user_ratings_test):
    adj_mat = sp.dok_matrix((n_users + n_items, n_users + n_items), dtype=np.float32)
    adj_mat = adj_mat.tolil()
    R = sp.dok_matrix((n_users, n_items), dtype=np.float32)
    for uid in user_ratings.keys():
        for item in user_ratings[uid]:
            if not item == user_ratings_test[uid]:
                R[uid, item] = 1
    R = R.tolil()

    adj_mat[:n_users, n_users:] = R
    adj_mat[n_users:, :n_users] = R.T
    adj_mat = adj_mat.todok()
    print('already create adjacency matrix', adj_mat.shape)

    
    def normalized_adj_single(adj):
        rowsum = np.array(adj.sum(1))
        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj = d_mat_inv.dot(adj)
        # norm_adj = adj.dot(d_mat_inv)
        print('generate single-normalized adjacency matrix.')
        return norm_adj.tocoo()

    def check_adj_if_equal(adj):
        dense_A = np.array(adj.todense())
        degree = np.sum(dense_A, axis=1, keepdims=False)
        temp = np.dot(np.diag(np.power(degree, -1)), dense_A)
        print('check normalized adjacency matrix whether equal to this laplacian matrix.')
        return temp

    norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
    mean_adj_mat = normalized_adj_single(adj_mat)

    print('already normalize adjacency matrix')
    return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()


def load_adjacency_list_data(adj_mat):
    tmp = adj_mat.tocoo()
    all_h_list = list(tmp.row)
    all_t_list = list(tmp.col)
    all_v_list = list(tmp.data)
    return all_h_list, all_t_list, all_v_list


def create_initial_A_values(n_factors, all_v_list):
    return np.array([all_v_list] * n_factors)


def generate_train_batch_for_all_overlap(user_ratings_1, user_ratings_test_1, n_1, user_ratings_2, user_ratings_test_2,
                                         n_2, batch_size):
    t_1 = []
    t_2 = []
    for b in range(batch_size):
        u = rd.sample(user_ratings_1.keys(), 1)[0]
        i_1 = rd.sample(user_ratings_1[u], 1)[0]
        i_2 = rd.sample(user_ratings_2[u], 1)[0]
        while i_1 == user_ratings_test_1[u]:
            i_1 = rd.sample(user_ratings_1[u], 1)[0]
        while i_2 == user_ratings_test_2[u]:
            i_2 = rd.sample(user_ratings_2[u], 1)[0]
        j_1 = rd.randint(0, n_1 - 1)
        j_2 = rd.randint(0, n_2 - 1)
        while j_1 in user_ratings_1[u]:
            j_1 = rd.randint(0, n_1 - 1)
        while j_2 in user_ratings_2[u]:
            j_2 = rd.randint(0, n_2 - 1)
        t_1.append([u, i_1, j_1])
        t_2.append([u, i_2, j_2])
    train_batch_1 = np.asarray(t_1)
    train_batch_2 = np.asarray(t_2)
    return train_batch_1, train_batch_2


def generate_test_batch_for_all_overlap(user_ratings_1, user_ratings_test_1, n_1,
                                        user_ratings_2, user_ratings_test_2, n_2):
    for u in user_ratings_1.keys():
        t_1 = []
        t_2 = []
        i_1 = user_ratings_test_1[u]
        i_2 = user_ratings_test_2[u]
        rated_1 = user_ratings_1[u]
        rated_2 = user_ratings_2[u]
        for j in range(999):
            k = np.random.randint(0, n_1-1)
            while k in rated_1:
                k = np.random.randint(0, n_1-1)
            t_1.append([u, i_1, k])
        for j in range(999):
            k = np.random.randint(0, n_2-1)
            while k in rated_2:
                k = np.random.randint(0, n_2-1)
            t_2.append([u, i_2, k])
        yield np.asarray(t_1), np.asarray(t_2)

def generate_test_batch_for_all_overlap_partial_users(user_ratings_1, user_ratings_test_1, n_1,
                                                      user_ratings_2, user_ratings_test_2, n_2, batch_size):
    for b in range(batch_size):
        t_1 = []
        t_2 = []
        u = rd.sample(user_ratings_1.keys(), 1)[0]
        i_1 = user_ratings_test_1[u]
        i_2 = user_ratings_test_2[u]
        rated_1 = user_ratings_1[u]
        rated_2 = user_ratings_2[u]
        for j in range(999):
            k = np.random.randint(0, n_1-1)
            while k in rated_1:
                k = np.random.randint(0, n_1-1)
            t_1.append([u, i_1, k])
        for j in range(999):
            k = np.random.randint(0, n_2-1)
            while k in rated_2:
                k = np.random.randint(0, n_2-1)
            t_2.append([u, i_2, k])
        yield np.asarray(t_1), np.asarray(t_2)