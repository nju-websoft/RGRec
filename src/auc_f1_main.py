import time
import torch
import os
import numpy as np
# from matplotlib import pyplot as plt
# from sklearn.metrics import roc_curve

from Args import args, args2str
from Graph import Graph
from RGRec_model import RGRec

device = "cuda" if torch.cuda.is_available() else "cpu"


def train_rgrec(rgrec, rule_id_list, model_file, eval_file_path, train_data_label, eval_data_label,
                test_data_label):
    lr = args.rkgcn_lr
    l2_weight = args.rkgcn_l2_weight
    n_epochs = args.rkgcn_n_epochs
    batch_size = args.rkgcn_batch_size

    optimizer = torch.optim.Adam(rgrec.parameters(), lr=lr, weight_decay=l2_weight)

    eval_auc_list = []

    test_auc_list = []
    test_f1_list = []

    eval_writer = open(eval_file_path, 'a+', encoding="UTF-8")
    eval_writer.write(args2str(args) + "\n")
    eval_writer.close()

    break_token = 0
    for epoch_i in range(n_epochs):
        print("Epoch: {}.".format(epoch_i))
        np.random.shuffle(train_data_label)
        start = 0
        # skip the last incomplete minibatch if its size < batch size
        while start + batch_size <= train_data_label.shape[0]:
            optimizer.zero_grad()
            loss = rgrec.update(train_data_label[start:start + args.rkgcn_batch_size, 0:2], rule_id_list,
                                      train_data_label[start:start + args.rkgcn_batch_size, 2])
            start += batch_size
            # print("Epoch: {}/{}, Start: {}/{}, Loss: {}"
            #       .format(epoch_i, args.n_epochs, start, train_data.shape[0], loss))
            optimizer.step()

        if (epoch_i + 1) % 3 != 0:
            continue

        print("Start evaluating.")
        # ctr_eval

        train_auc, train_f1, train_prec, train_reca = ctr_eval(epoch_i, rgrec,
                                                               train_data_label[:, 0:2],
                                                               train_data_label[:, 2],
                                                               rule_id_list,
                                                               args.rkgcn_batch_size)
        eval_auc, eval_f1, eval_prec, eval_reca = ctr_eval(epoch_i, rgrec,
                                                           eval_data_label[:, 0:2],
                                                           eval_data_label[:, 2],
                                                           rule_id_list,
                                                           args.rkgcn_batch_size)
        test_auc, test_f1, test_prec, test_reca = ctr_eval(epoch_i, rgrec,
                                                           test_data_label[:, 0:2],
                                                           test_data_label[:, 2],
                                                           rule_id_list,
                                                           args.rkgcn_batch_size)
        eval_auc_list.append(eval_auc)

        eval_str = 'epoch %d\ntrain auc: %.4f  f1: %.4f prec: %.4f reca: %.4f\n' \
                   'eval auc: %.4f  f1: %.4f prec: %.4f reca: %.4f\n' \
                   'test auc: %.4f  f1: %.4f prec: %.4f reca: %.4f' \
                   % (epoch_i, train_auc, train_f1, train_prec, train_reca,
                      eval_auc, eval_f1, eval_prec, eval_reca,
                      test_auc, test_f1, test_prec, test_reca)
        print(eval_str)

        if len(eval_auc_list) == 1 or eval_auc_list[-1] > eval_auc_list[-2]:
            test_auc_list.append(test_auc)
            test_f1_list.append(test_f1)
            break_token = 0
            rgrec.save_model(model_file)

            eval_writer = open(eval_file_path, 'a+', encoding="UTF-8")
            eval_writer.write("\n" + model_file + "\n")
            eval_writer.write(eval_str + "\n")
            eval_writer.close()

        if len(eval_auc_list) != 1 and eval_auc_list[-1] <= eval_auc_list[-2]:
            break_token += 1
            if break_token >= 3:
                break

    return test_auc_list[-1], test_f1_list[-1]


def ctr_eval(epoch_i, model, data, label, rule_id_list, batch_size):
    start = 0
    auc_list = []
    f1_list = []
    precision_list = []
    recall_list = []
    score_list = []
    label_list = []
    while start + batch_size <= data.shape[0]:
        auc, f1, prec, reca, score = model.auc_f1_prec_recal_scores_eval(data[start:start + batch_size], rule_id_list,
                                                                         label[start:start + batch_size])
        auc_list.append(auc)
        f1_list.append(f1)
        precision_list.append(prec)
        recall_list.append(reca)
        label_list.extend(list(label[start:start + batch_size]))
        score_list.extend(list(score))

        start += batch_size

    # fpr, tpr, thresholds = roc_curve(y_true=label_list, y_score=score_list, pos_label=1)

    # plt.plot([0, 1], [0, 1], linestyle='--')
    # plot the roc curve for the model
    # plt.plot(fpr, tpr, marker='.')
    # show the plot
    # plt.show()
    # plt.savefig('../data/music/pic/{}.png'.format(epoch_i))
    return float(np.mean(auc_list)), float(np.mean(f1_list)), float(np.mean(precision_list)), float(
        np.mean(recall_list))


def resample_train_eval_test():
    train_data = np.load(args.train_file)
    eval_data = np.load(args.eval_file)
    test_data = np.load(args.test_file)

    data_all = np.concatenate((train_data, eval_data, test_data), axis=0)

    rating_np_indices = np.array(range(len(data_all)))

    np.random.shuffle(rating_np_indices)
    train_indices, validate_indices, test_indices = np.split(rating_np_indices,
                                                             [int(.6 * len(rating_np_indices)),
                                                              int(.8 * len(rating_np_indices))])

    return data_all[train_indices, :], data_all[validate_indices, :], data_all[test_indices, :]


def topK_eval(model, data, rule_id_list, batch_size, user_nega_dict):
    data = data[np.where(data[:, 2] == 1)]

    item_list_to_eval = []
    user_list_to_eval = []
    start = 0

    hits5 = 0
    hits10 = 0
    hits15 = 0

    ndcg5 = 0
    ndcg10 = 0
    ndcg15 = 0
    for one_data in data:
        user_id = one_data[0]
        item_id = one_data[1]
        nega_item_id_list = list(user_nega_dict[user_id])
        nega_item_id_list.append(item_id)
        item_list_to_eval.append(nega_item_id_list)

        user_list_to_eval.append(user_id)

    while start + batch_size <= len(user_list_to_eval):
        t_hits5, t_hits10, t_hits15, t_ndcg5, t_ndcg10, t_ndcg15 = \
            model.hits_K_eval(user_list_to_eval[start:start + batch_size],
                              item_list_to_eval[start:start + batch_size],
                              rule_id_list)
        start += batch_size

        hits5 += t_hits5
        hits10 += t_hits10
        hits15 += t_hits15

        ndcg5 += t_ndcg5
        ndcg10 += t_ndcg10
        ndcg15 += t_ndcg15

    validate_user_num = len(user_list_to_eval)
    return float(hits5) / validate_user_num, float(hits10) / validate_user_num, float(hits15) / validate_user_num, \
           float(ndcg5) / validate_user_num, float(ndcg10) / validate_user_num, float(ndcg15) / validate_user_num


def process_main():
    retrain_rgrec = True

    g = Graph(args)
    g.load_e_r_mapping()
    g.construct_kg()
    g.construct_adj_e_id()

    rule_list = g.get_rules(g.r_name2id['interact'])
    rule_list.sort(key=lambda x: x[0])
    rule_id_list = [rule for _, _, rule, _ in rule_list]

    rkgcn = RGRec(args, len(g.e_id2name), len(g.r_id2name), len(rule_id_list), g.adj_e_id_by_r_id).to(device)

    time_stamp = str(time.time())
    time_stamp = time_stamp.replace(".", "")

    if not retrain_rgrec:
        print("Load latest model.")
        dir_list = []
        for dirs in os.listdir(args.rkgcn_model_root_path):
            if "." not in dirs:
                dir_list.append(dirs)

        dir_list.sort(reverse=True)

        loaded_model_path = args.rkgcn_model_root_path + dir_list[0] + "/" + "rkgcn_model.tar"

        rkgcn.load_model(loaded_model_path)

    model_folder = "{}{}/".format(args.rkgcn_model_root_path, time_stamp)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    eval_res_file_path = "{}{}".format(model_folder, "eval.txt")
    model_file_path = "{}{}".format(model_folder, "rkgcn_model.tar")

    train_data, eval_data, test_data = resample_train_eval_test()

    resample_train_data_file = "{}{}".format(model_folder, "train.npy")
    resample_eval_data_file = "{}{}".format(model_folder, "eval.npy")
    resample_test_data_file = "{}{}".format(model_folder, "test.npy")

    np.save(resample_train_data_file, train_data)
    np.save(resample_eval_data_file, eval_data)
    np.save(resample_test_data_file, test_data)

    print("Start Training.")
    test_auc, test_f1 = train_rgrec(rkgcn, rule_id_list, model_file_path, eval_res_file_path, train_data, eval_data,
                                    test_data)
    return test_auc, test_f1


def evaluate_model():
    g = Graph(args)
    g.load_e_r_mapping()
    g.construct_kg()
    g.construct_adj_e_id()

    rule_list = g.get_rules(g.r_name2id['interact'])
    rule_list.sort(key=lambda x: x[0])
    rule_id_list = [rule for _, _, rule, _ in rule_list]

    rkgcn = RGRec(args, len(g.e_id2name), len(g.r_id2name), len(rule_id_list), g.adj_e_id_by_r_id).to(device)

    print("Load latest model.")

    dir_list = []
    for dirs in os.listdir(args.rkgcn_model_root_path):
        if "." not in dirs and dirs != "backup":
            dir_list.append(dirs)

    dir_list.sort(reverse=True)

    loaded_model_folder = args.rkgcn_model_root_path + dir_list[-1] + "/"

    loaded_model_path = loaded_model_folder + "rkgcn_model.tar"

    rkgcn.load_model(loaded_model_path)

    train_data = np.load(loaded_model_folder + "train.npy")
    eval_data = np.load(loaded_model_folder + "eval.npy")
    test_data = np.load(loaded_model_folder + "test.npy")

    eval_res_file = loaded_model_folder + "new_eval.txt"

    train_rgrec(rkgcn, rule_id_list, loaded_model_path, eval_res_file, train_data, eval_data, test_data)


if __name__ == "__main__":
    # evaluate_model()
    auc_list = []
    f1_list = []
    for i in range(5):
        print("Start experiment {}.".format(i + 1))
        t_auc, t_f1 = process_main()
        auc_list.append(t_auc)
        f1_list.append(t_f1)
    print("AUC: {}, F1: {}".format(np.mean(auc_list), np.mean(f1_list)))
