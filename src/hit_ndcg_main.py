import time
import torch
import os
import numpy as np

from Args import args, args2str, DATASET
from Graph import Graph
from RGRec_model import RGRec

device = "cuda" if torch.cuda.is_available() else "cpu"

test_sample_folder = {}

test_sample_folder["movie_1M_30"] = ["15856167705488741",
                                     "15856129207194457",
                                     "15855930409478912",
                                     "1585499171588716",
                                     "1585574732984896"]

def get_user_nega_data_dict(file_path):
    print("Load user nega dict from {}".format(file_path))
    user_record_dict = {}
    with open(file_path, 'r', encoding="UTF-8") as f:
        for line in f.readlines():
            line_array = [int(one_id) for one_id in line.split()]
            if line_array[0] not in user_record_dict:
                user_record_dict[line_array[0]] = set()
            user_record_dict[line_array[0]] = set(line_array[1:])
    return user_record_dict


# def train_rkgcn(rkgcn_model, rule_id_list, model_file, eval_file_path, train_data_label, eval_data_label,
#                 test_data_label):
#     lr = args.rkgcn_lr
#     l2_weight = args.rkgcn_l2_weight
#     n_epochs = args.rkgcn_n_epochs
#     batch_size = args.rkgcn_batch_size
#
#     # hitsK eval
#     user_nega_dict = get_user_nega_data_dict(args.user_nega_data_file)
#
#     optimizer = torch.optim.Adam(rkgcn_model.parameters(), lr=lr, weight_decay=l2_weight)
#
#     eval_hits5_list = []
#     eval_hits10_list = []
#     eval_hits15_list = []
#
#     eval_ndcg5_list = []
#     eval_ndcg10_list = []
#     eval_ndcg15_list = []
#
#     eval_writer = open(eval_file_path, 'a+', encoding="UTF-8")
#     eval_writer.write(args2str(args) + "\n")
#     eval_writer.close()
#
#     break_token = 0
#     for epoch_i in range(n_epochs):
#         print("Epoch: {}.".format(epoch_i))
#         np.random.shuffle(train_data_label)
#         start = 0
#         # skip the last incomplete minibatch if its size < batch size
#         while start + batch_size <= train_data_label.shape[0]:
#             optimizer.zero_grad()
#             loss = rkgcn_model.update(train_data_label[start:start + args.rkgcn_batch_size, 0:2], rule_id_list,
#                                       train_data_label[start:start + args.rkgcn_batch_size, 2])
#             start += batch_size
#             # print("Epoch: {}/{}, Start: {}/{}, Loss: {}"
#             #       .format(epoch_i, args.n_epochs, start, train_data.shape[0], loss))
#             optimizer.step()
#
#         if (epoch_i + 1) % 3 != 0:
#             continue
#
#         print("Start evaluating.")
#
#         # hitsK eval
#
#         train_hits5, train_hits10, train_hits15, train_ndcg5, train_ndcg10, train_ndcg15 \
#             = topK_eval(rkgcn_model, train_data_label,
#                         rule_id_list, args.rkgcn_batch_size, user_nega_dict)
#
#         eval_hits5, eval_hits10, eval_hits15, eval_ndcg5, eval_ndcg10, eval_ndcg15 \
#             = topK_eval(rkgcn_model, eval_data_label,
#                         rule_id_list, args.rkgcn_batch_size, user_nega_dict)
#
#         test_hits5, test_hits10, test_hits15, test_ndcg5, test_ndcg10, test_ndcg15 \
#             = topK_eval(rkgcn_model, test_data_label,
#                         rule_id_list, args.rkgcn_batch_size, user_nega_dict)
#
#         eval_hits5_list.append(eval_hits5)
#         eval_hits10_list.append(eval_hits10)
#         eval_hits15_list.append(eval_hits15)
#
#         eval_ndcg5_list.append(eval_ndcg5)
#         eval_ndcg10_list.append(eval_ndcg10)
#         eval_ndcg15_list.append(eval_ndcg15)
#
#         top_eval_str = 'epoch %d\ntrain hits5: %.4f  hits10: %.4f hist15: %.4f\n' \
#                        'eval hits5: %.4f  hits10: %.4f hits15: %.4f \n' \
#                        'test hits5: %.4f  hits10: %.4f hits15: %.4f \n' \
#                        'train ndcg5: %.4f  ndcg10: %.4f ndcg15: %.4f\n' \
#                        'eval ndcg5: %.4f  ndcg10: %.4f ndcg15: %.4f \n' \
#                        'test ndcg5: %.4f  ndcg10: %.4f ndcg15: %.4f \n' \
#                        % (epoch_i, train_hits5, train_hits10, train_hits15,
#                           eval_hits5, eval_hits10, eval_hits15,
#                           test_hits5, test_hits10, test_hits15,
#                           train_ndcg5, train_ndcg10, train_ndcg15,
#                           eval_ndcg5, eval_ndcg10, eval_ndcg15,
#                           test_ndcg5, test_ndcg10, test_ndcg15)
#
#         print(top_eval_str)
#
#         if len(eval_hits15_list) == 1 or eval_hits15_list[-1] > eval_hits15_list[-2]:
#             rkgcn_model.save_model(model_file)
#
#             eval_writer = open(eval_file_path, 'a+', encoding="UTF-8")
#             eval_writer.write("\n" + model_file + "\n")
#             eval_writer.write(top_eval_str + "\n")
#             eval_writer.close()
#
#         if len(eval_hits15_list) != 1 and eval_hits15_list[-1] <= eval_hits15_list[-2]:
#             break_token += 1
#             if break_token >= 3:
#                 break


# def ctr_eval(epoch_i, model, data, label, rule_id_list, batch_size):
#     start = 0
#     auc_list = []
#     f1_list = []
#     precision_list = []
#     recall_list = []
#     score_list = []
#     label_list = []
#     while start + batch_size <= data.shape[0]:
#         auc, f1, prec, reca, score = model.auc_f1_prec_recal_scores_eval(data[start:start + batch_size], rule_id_list,
#                                                                          label[start:start + batch_size])
#         auc_list.append(auc)
#         f1_list.append(f1)
#         precision_list.append(prec)
#         recall_list.append(reca)
#         label_list.extend(list(label[start:start + batch_size]))
#         score_list.extend(list(score))
#
#         start += batch_size
#
#     return float(np.mean(auc_list)), float(np.mean(f1_list)), float(np.mean(precision_list)), float(
#         np.mean(recall_list))


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

    hitsK_dict = {}
    for i in range(15):
        hitsK_dict[i + 1] = 0

    ndcgK_dict = {}
    for i in range(15):
        ndcgK_dict[i + 1] = 0

    for one_data in data:
        user_id = one_data[0]
        item_id = one_data[1]
        nega_item_id_list = list(user_nega_dict[user_id])
        nega_item_id_list.append(item_id)
        item_list_to_eval.append(nega_item_id_list)

        user_list_to_eval.append(user_id)

    while start + batch_size <= len(user_list_to_eval):
        t_hitK_dict, t_ndcg_dict = \
            model.hits_K_eval(user_list_to_eval[start:start + batch_size],
                              item_list_to_eval[start:start + batch_size],
                              rule_id_list)
        start += batch_size

        for i in range(15):
            hitsK_dict[i + 1] += t_hitK_dict[i + 1]

        for i in range(15):
            ndcgK_dict[i + 1] += t_ndcg_dict[i + 1]

    validate_user_num = len(user_list_to_eval)

    for i in range(15):
        hitsK_dict[i + 1] /= validate_user_num

    for i in range(15):
        ndcgK_dict[i + 1] /= validate_user_num

    hit_ndcg_str = ""
    for i in range(15):
        hit_ndcg_str = "{}\thit{}: {}".format(hit_ndcg_str, i + 1, hitsK_dict[i + 1])
    hit_ndcg_str += "\n"
    for i in range(15):
        hit_ndcg_str = "{}\tndcg{}: {}".format(hit_ndcg_str, i + 1, ndcgK_dict[i + 1])
    hit_ndcg_str += "\n"

    return hit_ndcg_str, hitsK_dict, ndcgK_dict


#
# def process_main():
#     retrain_rkgcn = True
#
#     g = Graph(args)
#     g.load_e_r_mapping()
#     g.construct_kg()
#     g.construct_adj_e_id()
#
#     rule_list = g.get_rules(g.r_name2id['interact'])
#     rule_list.sort(key=lambda x: x[0])
#     rule_id_list = [rule for _, _, rule, _ in rule_list]
#
#     rkgcn = RKGCN(args, len(g.e_id2name), len(g.r_id2name), len(rule_id_list), g.adj_e_id_by_r_id).to(device)
#
#     time_stamp = str(time.time())
#     time_stamp = time_stamp.replace(".", "")
#
#     if not retrain_rkgcn:
#         print("Load latest model.")
#         dir_list = []
#         for dirs in os.listdir(args.rkgcn_model_root_path):
#             if "." not in dirs:
#                 dir_list.append(dirs)
#
#         dir_list.sort(reverse=True)
#
#         loaded_model_path = args.rkgcn_model_root_path + dir_list[0] + "/" + "rkgcn_model.tar"
#
#         rkgcn.load_model(loaded_model_path)
#
#     model_folder = "{}{}/".format(args.rkgcn_model_root_path, time_stamp)
#     if not os.path.exists(model_folder):
#         os.makedirs(model_folder)
#
#     eval_res_file_path = "{}{}".format(model_folder, "eval.txt")
#     model_file_path = "{}{}".format(model_folder, "rkgcn_model.tar")
#
#     train_data, eval_data, test_data = resample_train_eval_test()
#
#     resample_train_data_file = "{}{}".format(model_folder, "train.npy")
#     resample_eval_data_file = "{}{}".format(model_folder, "eval.npy")
#     resample_test_data_file = "{}{}".format(model_folder, "test.npy")
#
#     np.save(resample_train_data_file, train_data)
#     np.save(resample_eval_data_file, eval_data)
#     np.save(resample_test_data_file, test_data)
#
#     print("Start Training.")
#     train_rkgcn(rkgcn, rule_id_list, model_file_path, eval_res_file_path, train_data, eval_data, test_data)
#     print("Finish Training.")


def only_evaluate(rkgcn_model, rule_id_list, test_data_label):
    batch_size = args.rkgcn_batch_size

    # hitsK eval
    user_nega_dict = get_user_nega_data_dict(args.user_nega_data_file)

    tmp_hit_ndcg_str, tmp_hitK_dict, tmp_ndcg_dict = topK_eval(rkgcn_model, test_data_label, rule_id_list, batch_size,
                                                               user_nega_dict)

    print(tmp_hit_ndcg_str)
    return tmp_hit_ndcg_str, tmp_hitK_dict, tmp_ndcg_dict


def evaluate_model(loaded_model_folder):
    g = Graph(args)
    g.load_e_r_mapping()
    g.construct_kg()
    g.construct_adj_e_id()

    rule_list = g.get_rules(g.r_name2id['interact'])
    rule_list.sort(key=lambda x: x[0])
    rule_id_list = [rule for _, _, rule, _ in rule_list]

    rkgcn = RGRec(args, len(g.e_id2name), len(g.r_id2name), len(rule_id_list), g.adj_e_id_by_r_id).to(device)

    # print("Load latest model.")
    # dir_list = []
    # for dirs in os.listdir(args.rkgcn_model_root_path):
    #     if "." not in dirs and dirs != "backup":
    #         dir_list.append(dirs)
    # dir_list.sort(reverse=True)
    # loaded_model_folder = args.rkgcn_model_root_path + dir_list[-1] + "/"
    # loaded_model_path = loaded_model_folder + "rkgcn_model.tar"

    loaded_model_path = loaded_model_folder + "rkgcn_model.tar"

    rkgcn.load_model(loaded_model_path)
    rkgcn.device_num = 0

    test_data = np.load(loaded_model_folder + "test.npy")

    return only_evaluate(rkgcn, rule_id_list, test_data)


def batch_reeval():
    hitsK_dict = {}
    for i in range(15):
        hitsK_dict[i + 1] = 0

    ndcgK_dict = {}
    for i in range(15):
        ndcgK_dict[i + 1] = 0

    model_folder = args.rkgcn_model_root_path
    for one_model in test_sample_folder[DATASET]:
        model_file = "{}{}/".format(model_folder, one_model)
        print("Model file {}".format(model_file))
        hit_ndcg_str, tmp_hitK_dict, tmp_ndcg_hit_dict = evaluate_model(model_file)

        for i in range(15):
            ndcgK_dict[i + 1] += tmp_ndcg_hit_dict[i + 1]
            hitsK_dict[i + 1] += tmp_hitK_dict[i + 1]

        hitsk_eval_result_file = "{}{}/hits_eval.txt".format(model_folder, one_model)
        with open(hitsk_eval_result_file, 'w', encoding="UTF-8") as f:
            f.write("{}".format(hit_ndcg_str))

    sample_num = len(test_sample_folder[DATASET])
    average_str = ""
    for i in range(15):
        average_str = "{}\thit{}: {}".format(average_str, i + 1, hitsK_dict[i + 1] / sample_num)
    average_str += "\n"
    for i in range(15):
        average_str = "{}\tndcg{}: {}".format(average_str, i + 1, ndcgK_dict[i + 1] / sample_num)
    average_str += "\n"

    print(average_str)


if __name__ == "__main__":
    # evaluate_model()
    # for i in range(3):
    #     print("Start experiment {}.".format(i + 1))
    #     process_main()
    batch_reeval()
