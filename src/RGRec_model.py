import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, roc_curve
import math
from ALogger import ALogger

device = "cuda" if torch.cuda.is_available() else "cpu"


class RGRec(nn.Module):
    def __init__(self, args, e_num, r_num, rule_size, adj_e_id_by_r_id_dict):
        super(RGRec, self).__init__()

        self.logger = ALogger("PRA", True).getLogger()
        self.device_num = 0

        self.e_num = e_num
        self.r_num = r_num
        self.rule_size = rule_size
        self.adj_e_id_by_r_id_dict = adj_e_id_by_r_id_dict
        self._build_param(args)
        self._build_model()

    def _build_param(self, args):
        torch.cuda.set_device(self.device_num)
        self.logger.info("Device: {}".format(torch.cuda.current_device()))
        self.dim = args.rkgcn_dim
        self.batch_size = args.rkgcn_batch_size
        self.max_step = args.rkgcn_max_step
        self.neighbour_size = args.rkgcn_neighbour_size
        self.dropout = args.rkgcn_dropout

        self.rule_weight_file_path = args.rule_weight_file_path
        self.pre_train_rule_weight = args.pre_train_rule_weight
        self.freeze_rule_weight = args.freeze_rule_weight

        self.rkgcn_model_file_path = args.rkgcn_model_root_path

        self.aggregate_name = args.aggregator_name

    def _build_model(self):
        self.ent_embed = nn.Embedding(self.e_num, self.dim, max_norm=1.0).to(device)
        torch.nn.init.xavier_uniform(self.ent_embed.weight)

        if self.pre_train_rule_weight:
            self.logger.info("Use pretrained rule weight.")
            loaded_rule_weight = np.load(self.rule_weight_file_path)
            self.rule_embed = nn.Embedding.from_pretrained(torch.FloatTensor(loaded_rule_weight),
                                                           freeze=self.freeze_rule_weight)
            # self.rule_embed = nn.Embedding.from_pretrained(torch.FloatTensor(loaded_rule_weight),
            #                                                freeze=self.freeze_rule_weight,
            #                                                max_norm=1.0)
        else:
            self.logger.info("Use raw pretrained rule weight.")
            self.rule_embed = nn.Embedding(1, self.rule_size, max_norm=1.0).to(device)
            torch.nn.init.xavier_uniform(self.rule_embed.weight)
            # self.rule_embed = torch.Tensor(np.ones([1, self.rule_size]) / self.rule_size).to(device)

        # self.aggregate_layer = nn.Linear(self.dim, self.dim).to(device)
        self.concat_aggregate_layer = nn.Linear(self.dim * 2, self.dim).to(device)

        # torch.nn.init.xavier_uniform(self.aggregate_layer.weight)
        torch.nn.init.xavier_uniform(self.concat_aggregate_layer.weight)

    def save_model(self, model_file_path):
        self.logger.info("Save rkgcn model to {}.".format(model_file_path))

        # torch.save(self.state_dict(), self.rkgcn_model_file_path)
        torch.save(self.state_dict(), model_file_path)

    def load_model(self, model_file_path):
        self.logger.info("Load rkgcn model from {}.".format(model_file_path))
        self.load_state_dict(torch.load(model_file_path, map_location=device))

    def load_rule_weight(self, rule_weight_file_path):
        return np.load(rule_weight_file_path)

    def user_rep(self, user_array, rule_list):
        adj_e_list = []

        for one_rule in rule_list:
            # entities, list, [i,[batch_size,neighbour_size^i]], i in [1,...,max_step]
            adj_e = self.get_adj_ent_of_r_id(one_rule, user_array)
            adj_e_list.append(adj_e)

        res_list = []
        for step_i in range(self.max_step + 1):
            one_step_node = []
            for rule_idx in range(len(rule_list)):
                one_step_node.append(adj_e_list[rule_idx][step_i])
            # np.hstack(one_step_node), [batch_size,rule_num * neighbour^step_i]
            res_list.append(np.hstack(one_step_node))

        # res, [batch_size, rule_num, dim]
        res = self.aggregate(res_list)

        # output, [batch_size,]
        output = torch.sum(
            res.to(device) * self.rule_embed(torch.LongTensor([0]).to(device)).view((self.rule_size, 1)).to(device),
            dim=-2)

        return output

    def update(self, user_item_array, rule_list, label_array):

        output = self.user_rep(user_item_array[:, 0], rule_list)
        item_list = np.array(user_item_array)[:, 1]
        item_embed = self.ent_embed(torch.LongTensor(item_list).to(device))
        res_prob = torch.sigmoid(torch.sum(output * item_embed, dim=-1)).to(device)

        criterion = nn.BCELoss().to(device)

        loss = criterion(res_prob, torch.Tensor(label_array).to(device))
        loss.backward()
        return loss.cpu().item() / self.batch_size

    def get_adj_ent_of_r_id(self, one_rule, seed):
        # seed, np array, [batch_size,1]
        # entities, list, [np array], [[batch_size,1]]
        entities = [seed]
        for idx in range(self.max_step):
            if idx >= len(one_rule):
                r_id = self.r_num
            else:
                r_id = one_rule[idx]
            # neighbours, np array, [e_num, neighbour_size]
            neighbours = self.adj_e_id_by_r_id_dict[r_id]
            # r_neighbour, np array, [self.batch_size,-1]
            # entities[idx], np_array, [batch_size, neighbour_size^idx]
            r_neighbour = neighbours[entities[idx], :].reshape([self.batch_size, -1])
            entities.append(r_neighbour)
        # entities, list, [i,[batch_size,neighbour_size^i]], i in [0,1,...,max_step]
        return entities

    def mix_neighbour_vectors(self, neighbour_vectors):
        # neighbor_vectors, [batch_size, -1, neighbor_size, dim]
        # [batch_size, -1, dim]
        neighbors_aggregated = torch.mean(neighbour_vectors, dim=2)

        return neighbors_aggregated

    # self_vectors, [batch_size,-1,dim]
    # neighbor_vectors, [batch_size,-1,neighbour_size,dim]
    # def sum_aggregator(self, self_vectors, neighbor_vectors, act):
    #     # [batch_size, -1, dim]
    #     neighbors_agg = self.mix_neighbour_vectors(neighbor_vectors)
    #
    #     # [-1, dim]
    #     output = (self_vectors + neighbors_agg).view([-1, self.dim])
    #     output = F.dropout(output, p=self.dropout)
    #     output = self.aggregate_layer(output)
    #
    #     # [batch_size, -1, dim]
    #     output = output.view([self.batch_size, -1, self.dim]).to(device)
    #
    #     return act(output)

    def concat_aggreator(self, self_vectors, neighbor_vectors, act):
        # [batch_size, -1, dim]
        neighbors_agg = self.mix_neighbour_vectors(neighbor_vectors)

        # [-1, dim]
        output = torch.cat((self_vectors, neighbors_agg), -1)
        output = output.view([-1, self.dim * 2])
        output = F.dropout(output, p=self.dropout)
        output = self.concat_aggregate_layer(output)

        # [batch_size, -1, dim]
        output = output.view([self.batch_size, -1, self.dim]).to(device)

        return act(output)

    # def neighbor_aggregator(self, self_vectors, neighbor_vectors, act):
    #     # [batch_size, -1, dim]
    #     neighbors_agg = self.mix_neighbour_vectors(neighbor_vectors)
    #
    #     output = neighbors_agg.view([-1, self.dim])
    #     output = F.dropout(output, p=self.dropout)
    #     output = self.aggregate_layer(output)
    #
    #     # [batch_size, -1, dim]
    #     output = output.view([self.batch_size, -1, self.dim]).to(device)
    #
    #     return act(output)

    def aggregate(self, entities):

        entity_vectors = [self.ent_embed(torch.LongTensor(i).to(device)).to(device) for i in entities]

        for i in range(self.max_step):
            act = torch.tanh if i == self.max_step - 1 else torch.relu
            entity_vectors_next_iter = []
            for hop in range(self.max_step - i):
                shape = [self.batch_size, -1, self.neighbour_size, self.dim]
                vector = getattr(self, self.aggregate_name)(
                    self_vectors=entity_vectors[hop].view([self.batch_size, -1, self.dim]),
                    neighbor_vectors=entity_vectors[hop + 1].view(shape), act=act)
                entity_vectors_next_iter.append(vector)
            entity_vectors = entity_vectors_next_iter

        res = entity_vectors[0].view([self.batch_size, -1, self.dim])
        return res

    def auc_f1_prec_recal_scores_eval(self, user_item_array, rule_list, label_array):
        output = self.user_rep(user_item_array[:, 0], rule_list)
        item_list = np.array(user_item_array)[:, 1]
        item_embed = self.ent_embed(torch.LongTensor(item_list).to(device))
        scores = torch.sigmoid(torch.sum(output * item_embed, dim=-1)).cpu().detach().numpy()

        auc = roc_auc_score(y_true=label_array, y_score=scores)
        scores[scores >= 0.5] = 1
        scores[scores < 0.5] = 0

        precision = precision_score(y_true=label_array, y_pred=scores)
        recall = recall_score(y_true=label_array, y_pred=scores)
        f1 = f1_score(y_true=label_array, y_pred=scores)
        return auc, f1, precision, recall, scores

    def cal_ndcg(self, prediction_item, target_item):
        for idx, item in enumerate(prediction_item):
            if item == target_item:
                return math.log(2) / math.log(idx + 2)

        return 0

    def hits_K_eval(self, user_list, item_list, rule_list):
        output = self.user_rep(user_list, rule_list)
        output = output.view((len(user_list), 1, -1))

        item_embed = self.ent_embed(torch.LongTensor(item_list).to(device))
        item_embed_transposed = torch.transpose(item_embed, 2, 1)
        scores = torch.bmm(output, item_embed_transposed).view(len(user_list), -1).cpu().detach().numpy()

        ordered_scores = np.argsort(scores, axis=-1)
        ordered_scores = ordered_scores.transpose()[::-1].transpose()

        hitsK_dict = {}
        for i in range(15):
            hitsK_dict[i + 1] = 0

        ndcgK_dict = {}
        for i in range(15):
            ndcgK_dict[i + 1] = 0

        for idx, user_id in enumerate(user_list):

            for i in range(15):
                hits_k = i + 1
                if 100 in list(ordered_scores[idx][:hits_k]):
                    hitsK_dict[hits_k] += 1

                ndcgK_dict[hits_k] += self.cal_ndcg(ordered_scores[idx][:hits_k], 100)

        return hitsK_dict, ndcgK_dict
