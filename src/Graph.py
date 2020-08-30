import numpy as np
import os
import random
import argparse
import scipy
from scipy.stats import chi2_contingency

from Args import args


class Graph:
    def __init__(self, g_args):
        self._build_param(g_args)

        """
        {
            r_name: r_id,
            r_name: r_id,
            ...
        } 
        """
        self.r_name2id = dict()

        """
        {
            r_id: r_name,
            r_id: r_name,
            ...
        } 
        """
        self.r_id2name = dict()

        """
        {
            e_id: e_name,
            e_id: e_name,
            ...
        } 
        """
        self.e_id2name = dict()

        """
        {
            e_name: e_id,
            e_name: e_id,
            ...
        } 
        """
        self.e_name2id = dict()

        """
        {
            h_id: 
                {
                    r_id:[t_id,...,t_id],
                    r_id:[t_id,...,t_id],
                    ...
                },
            ...
        }
        """
        self.kg = dict()

        """
        {
            r_id:[[h_id,t_id],...,[h_id,t_id]],
            r_id:[[h_id,t_id],...,[h_id,t_id]],
            ...
        }
        """
        self.r_id2ht_id = dict()

        self.adj_e_id_by_r_id = dict()

    def _build_param(self, args):

        self.none_node_name = "None Node"
        self.none_node_id = -1

        self.neighbour_size = args.rkgcn_neighbour_size
        self.entity_file = args.entity_file
        self.relation_file = args.relation_file
        self.kg_file = args.kg_file
        self.rule_file = args.rule_file
        self.train_file = args.train_file

        self.filter_inv_pattern = args.filter_inv_pattern
        self.rule_seg = args.rule_seg
        self.reserved_rule_num_by_frequency = args.reserved_rule_num_by_frequency

        self.max_step = args.rkgcn_max_step
        self.chi_thresh = args.chi_thresh

    def load_e_r_mapping(self):
        print("entity file: {}".format(self.entity_file))
        if len(self.e_id2name) == 0:
            print("load e  mapping")
            for line in open(self.entity_file, 'r', encoding="UTF-8").readlines()[1:]:
                line_array = line.split("\t")
                if len(line_array) > 2:
                    e_id = line_array[-1]
                    e_name_list = []
                    for one_name in line_array[:-1]:
                        e_name_list.append(one_name)
                    e_name = " ".join(e_name_list)
                else:
                    e_name, e_id = line.split("\t")
                e_id = int(e_id)
                self.e_name2id[e_name] = e_id
                self.e_id2name[e_id] = e_name

            self.none_node_id = len(self.e_id2name)
            self.e_id2name[self.none_node_id] = self.none_node_name
            self.e_name2id[self.none_node_name] = self.none_node_id

        print("relation file: {}".format(self.relation_file))
        if len(self.r_name2id) == 0:
            print("load r mapping")
            for line in open(self.relation_file, 'r', encoding="UTF-8").readlines()[1:]:
                r_name, r_id = line.split()
                r_id = int(r_id)
                self.r_name2id[r_name] = r_id
                self.r_id2name[r_id] = r_name

    def construct_kg(self):
        if len(self.kg) != 0:
            return

        print("construct kg")
        if os.path.exists(self.kg_file + '.npy'):
            kg_np = np.load(self.kg_file + '.npy')
        else:
            kg_np = np.loadtxt(self.kg_file + '.txt', dtype=np.int64)
            np.save(self.kg_file + '.npy', kg_np)

        kg_np = kg_np.astype(np.int32)
        for h_id, r_id, t_id in kg_np:
            if h_id not in self.kg:
                self.kg[h_id] = dict()
            if r_id not in self.kg[h_id]:
                self.kg[h_id][r_id] = []
            self.kg[h_id][r_id].append(t_id)

            if r_id not in self.r_id2ht_id:
                self.r_id2ht_id[r_id] = []
            self.r_id2ht_id[r_id].append([h_id, t_id])
        self.kg[self.none_node_id] = dict()

    def construct_adj_e_id(self):
        print("Construct_ajd_e_id")
        if len(self.adj_e_id_by_r_id) != 0:
            return

        self.load_e_r_mapping()
        self.construct_kg()
        e_num = len(self.e_id2name)
        print("e num: {}".format(e_num))

        for r_id in self.r_id2name:
            self.adj_e_id_by_r_id[r_id] = np.zeros(shape=[e_num, self.neighbour_size], dtype=np.int32)
            for e_id in self.e_id2name:
                r_neighbour_list = self.get_neighbours(e_id, r_id)
                if len(r_neighbour_list) >= self.neighbour_size:
                    sampled_neighbours = np.random.choice(r_neighbour_list, size=self.neighbour_size,
                                                          replace=False)
                else:
                    sampled_neighbours = np.random.choice(r_neighbour_list, size=self.neighbour_size,
                                                          replace=True)
                self.adj_e_id_by_r_id[r_id][e_id] = np.array(sampled_neighbours).astype(dtype=np.int32)

        r_num = len(self.r_id2name)
        self.adj_e_id_by_r_id[r_num] = np.ones(shape=[e_num, self.neighbour_size],
                                               dtype=np.int32) * self.none_node_id

    def get_neighbours(self, h_id, r_id):
        self.construct_kg()
        if r_id in self.kg[h_id]:
            return self.kg[h_id][r_id]
        else:
            return [self.none_node_id]

    """
    Search from head to get paths whose length are under max_step,
    for example, when step is 3, the length of paths got are 1,2,3
    Parameters:
    -----------
    start_id: int, entity_id, index of start point
    max_step: int, max length of path

    Returns:
    -----------
    out:  list, [[-1,e_id,r_id...,e_id],[],...], the list of path for searched.
    """

    def search_unidirect(self, start_id, max_step):
        res = []
        res.append([[-1, start_id]])
        current_path_list = [[-1, start_id]]
        for step_i in range(max_step):
            temp_path = []
            for one_path in current_path_list:
                current_node = self.kg[one_path[-1]]
                for relation_id in current_node:
                    for entity_id in current_node[relation_id]:
                        if step_i >= 1 and entity_id == one_path[-3]:
                            continue
                        temp_n = one_path.copy()
                        temp_n.append(relation_id)
                        temp_n.append(entity_id)
                        temp_path.append(temp_n)
            res.append(temp_path.copy())
            current_path_list = temp_path.copy()
        return res

    """
    Search from bidirection
    Parameters:
    -----------
    head_id: int, index of head entity
    tail_id: int, index of tail entity
    max_step: int, the max length between head and tail.
    for example, when step = 3, we get paths whose length is 1, 2 and 3

    Returns:
    -----------
    out: list, [[-1,e_id,r_id,e_id,...,-1],[],[],...]
    """

    def search_bidirect(self, head_id, tail_id, max_step):
        res = []
        left_path = self.search_unidirect(head_id, int(max_step / 2))
        right_path = self.search_unidirect(tail_id, max_step - int(max_step / 2))
        for step_i in range(max_step):
            left_len = int((step_i + 1) / 2)
            right_len = (step_i + 1) - left_len
            temp_res = self.join_left_and_right_path(left_path[left_len],
                                                     right_path[right_len])
            res.extend(temp_res)
        return res

    '''
    Connect two path found by unidirection search
    Parameters:
    -----------
    left_path: list, [[-1,e_id,r_id,...,e_id],[],...]
    right_path: list, [[-1,e_id,r_id,...,e_id],[],...]

    Returns:
    -----------
    out: list [[-1,e_id,r_id,...,e_id,-1],[],[],...]
    '''

    def join_left_and_right_path(self, left_path, right_path):
        res = []
        left_dict = {}
        right_dict = {}

        for l_p in left_path:
            end_point = l_p[-1]
            if end_point not in left_dict:
                left_dict[end_point] = []
            left_dict[end_point].append(l_p)

        for r_p in right_path:
            end_point = r_p[-1]
            if end_point not in right_dict:
                right_dict[end_point] = []
            right_dict[end_point].append(r_p)

        for key in left_dict.keys():
            if key in right_dict:
                for l_p in left_dict[key]:
                    for r_p in right_dict[key]:
                        temp_one_path = l_p[:-1]
                        for r_p_i in range(len(r_p) - 1, 0, -1):
                            if r_p_i % 2 != 0:
                                temp_one_path.append(r_p[r_p_i])
                            else:
                                relation_name = self.r_id2name[r_p[r_p_i]]
                                if relation_name.startswith("inv_"):
                                    relation_name = relation_name[4:]
                                else:
                                    relation_name = "inv_" + relation_name
                                temp_one_path.append(self.r_name2id[relation_name])
                        temp_one_path.append(-1)
                        res.append(temp_one_path)
        return res

    """
    Search paths that can conclude to r_id
    Parameters:
    -----------
    r_id: int, index of target relation(r_idx)
    max_step: int, max length of searched path
    train_data: list, [[h_id,t_id],[h_id,t_id],...,[h_id,t_id]], [h_id,t_id] used to search inference rules

    Returns:
    -----------
    out: list, list
    first list is the path of r, [[r_idx,r_idx,r_idx,...],...,[r_idx,r_idx,r_idx,...]]
    second list is the path of e and r, [[-1,e_idx,r_idx,e_idx,r_iex,...,-1],...,[-1,e_idx,r_idx,e_idx,r_iex,...,-1]]
    """

    def search_path(self, r_id, max_step, train_data):
        search_r_path_num = {}
        searched_r_path = {}
        searched_e_r_path = []

        print("Start Searching Path:\nR:{} Train Num:{}".format(self.r_id2name[r_id], len(train_data)))

        # train_data = random.sample(train_data, 1000)
        for idx, ht in enumerate(train_data):
            h_id = ht[0]
            t_id = ht[1]
            print('{}/{} Relation: {} H: {} T: {}'
                  .format(idx + 1, len(train_data),
                          self.r_id2name[r_id],
                          self.e_id2name[h_id], self.e_id2name[t_id]))
            path_found = self.search_bidirect(h_id, t_id, max_step)
            for p in path_found:
                r_path = self.extract_r_id_path(p)
                if len(r_path) == 1 and r_path[0] == r_id:
                    continue
                if self.filter_inv_pattern and self.has_inverse_r_in_r_path(r_path):
                    continue

                r_path_key = self.rule_seg.join(map(str, r_path))

                searched_e_r_path.append(p)
                if r_path_key not in searched_r_path:
                    searched_r_path[r_path_key] = r_path
                    search_r_path_num[r_path_key] = 1
                else:
                    search_r_path_num[r_path_key] += 1

        print("Total rule num: {}".format(len(search_r_path_num.keys())))

        res_r_id_path_list = []
        for key, value in list(
                sorted(
                    search_r_path_num.items(),
                    key=lambda d: d[1],
                    reverse=True))[:self.reserved_rule_num_by_frequency]:
            res_r_id_path_list.append(searched_r_path[key])
        return res_r_id_path_list, searched_e_r_path

    """
    Convert r_id to r_name
    Parameters:
    -----------
    r_id_path_list: list, [r_id,r_id,...]

    Returns:
    -----------
    out: list, [r_name,r_name,...]
    """

    def display_r_path(self, r_id_path):
        return [self.r_id2name[r_id] for r_id in r_id_path]

    """
    Extract r_path for e_r_path
    Parameters:
    -----------
    e_r_path: list [-1,e_id,r_id,e_id,...,-1]

    Returns:
    -----------
    out: list [r_id,r_id,...]
    """

    def extract_r_id_path(self, e_r_id_path):
        r_path = []
        for idx in range(1, len(e_r_id_path) - 1):
            if idx % 2 == 0:
                r_path.append(e_r_id_path[idx])
        return r_path

    '''
    Check if a r_path has inverse pattern relation
    Parameters:
    -----------
    r_path: list, [r_idx,r_idx,...]
    a list of relation path

    Returns:
    -----------
    out: boolean
    True if r_path has inverse pattern relation, false otherwish.
    '''

    def has_inverse_r_in_r_path(self, r_path):
        for r_path_i in range(len(r_path) - 1):
            if self.is_inverse_r_id(r_path[r_path_i], r_path[r_path_i + 1]):
                return True
        return False

    '''
    Check if two relation is inverse to each other.
    Parameters:
    -----------
    r_id1: int, index of relation1
    r_id2: int, index of relation2

    Returns:
    -----------
    out: boolean
    True if r_idx1 and r_idx2 is inverse to each other, false otherwise.
    '''

    def is_inverse_r_id(self, r_id1, r_id2):
        r_name1 = self.r_id2name[r_id1]
        r_name2 = self.r_id2name[r_id2]
        if r_name1.startswith("inv_") and r_name1 == "inv_{}".format(r_name2):
            return True
        if r_name2.startswith("inv_") and r_name2 == "inv_{}".format(r_name1):
            return True
        return False

    """
    Get a list of ht which can pass r_path
    Parameters:
    -----------
    r_path: list,
    a rule: [r_idx,r_idx,r_idx,...], for example: [2,3,6,...]

    Return:
    -----------
    out: list, [[h,t],[h,t],[h,t],...]
    list of passed ht
    """

    def get_passed_ht(self, r_path):
        # start_time = time.time()
        # def time_exceed():
        #     end_time = time.time()
        #     if check_time_for_get_passed_ht and (
        #             end_time - start_time) > time_limit_for_get_passed_ht:
        #         print("Elapsed: {}".format(end_time - start_time))
        #         return True

        if len(r_path) == 1:
            return self.r_id2ht_id[r_path[0]]

        left_path = self.r_id2ht_id[r_path[0]]
        left_step = 1
        inv_r_id = self.convert_r(r_path[-1])
        right_path = self.r_id2ht_id[inv_r_id]
        right_step = 1

        path_num_constrain = 100000
        if len(left_path) > path_num_constrain:
            left_path = random.sample(left_path, path_num_constrain)

        if len(right_path) > path_num_constrain:
            right_path = random.sample(right_path, path_num_constrain)

        while len(r_path) - (left_step + right_step) > 0:
            # print("Left Path length: {}.".format(len(left_path)))
            # print("Right Path length: {}.".format(len(right_path)))
            # if limit_branch_node_num and len(left_path) > branch_node_limit:
            #     left_path = random.sample(left_path, branch_node_limit)
            # if limit_branch_node_num and len(right_path) > branch_node_limit:
            #     right_path = random.sample(right_path, branch_node_limit)

            temp_left_path = []
            temp_right_path = []
            if len(left_path) < len(right_path):
                left_step += 1
                r_id = r_path[left_step - 1]
                for ht in left_path:
                    c_node = self.kg[ht[-1]]
                    if r_id not in c_node:
                        continue
                    for tail in c_node[r_id]:
                        # if time_exceed():
                        #     return True, []
                        temp_ht = ht.copy()
                        temp_ht.append(tail)
                        temp_left_path.append(temp_ht)
                left_path = temp_left_path
            else:
                right_step += 1
                r_id = r_path[-right_step]
                inv_r_id = self.convert_r(r_id)
                for ht in right_path:
                    c_node = self.kg[ht[-1]]
                    if inv_r_id not in c_node:
                        continue
                    for tail in c_node[inv_r_id]:
                        temp_ht = ht.copy()
                        temp_ht.append(tail)
                        temp_right_path.append(temp_ht)
                right_path = temp_right_path
        res = {}
        left_dict = {}
        for path in left_path:
            if path[-1] not in left_dict:
                left_dict[path[-1]] = []
            left_dict[path[-1]].append(path)
        for path in right_path:
            if path[-1] in left_dict:
                for l_p in left_dict[path[-1]]:
                    temp_token = "{};{}".format(l_p[0], path[0])
                    if temp_token not in res:
                        res[temp_token] = [l_p[0], path[0]]
        # print("Elapsed: {}".format(time.time() - start_time))
        return [res[key] for key in res]

    '''
    Convert relation(r_id) to its inverse relation(r_id)
    Parameters:
    -----------
    r_id: int, the index of relation

    Returns:
    -----------
    out: int, index of its inverse relation
    '''

    def convert_r(self, r_id):
        name = self.r_id2name[r_id]
        if name.startswith("inv_"):
            name = name[4:]
        else:
            name = "inv_" + name
        return self.r_name2id[name]

    '''
    Test the chi square of a rule
    Parameters:
    -----------
    r_id: int, the index of relation
    one_path: list, [r_id,r_id,..,r_id]

    Returns:
    -----------
    out: float, p-value
    '''

    def chi_square(self, r_id, one_id_path):
        print("Get chi square of: {}".format(self.display_r_path(one_id_path)))
        user_id_set = set()
        item_id_set = set()
        ground_truth_set = set()

        for h_id, t_id in self.r_id2ht_id[r_id]:
            user_id_set.add(h_id)
            item_id_set.add(t_id)
            ground_truth_set.add("{};{}".format(h_id, t_id))

        print("Get passed ht of {}".format(self.display_r_path(one_id_path)))

        passed_ht = self.get_passed_ht(one_id_path)
        passed_ht_token_final = set()
        for h_id, t_id in passed_ht:
            if h_id in user_id_set and t_id in item_id_set:
                passed_ht_token_final.add("{};{}".format(h_id, t_id))

        true_posi_num = len(ground_truth_set & passed_ht_token_final)
        false_posi_num = len(ground_truth_set - passed_ht_token_final)

        false_nege_num = len(passed_ht_token_final) - true_posi_num

        true_nege_num = len(user_id_set) * len(item_id_set) - true_posi_num - false_posi_num - false_nege_num

        ob = np.array([[true_posi_num, false_nege_num], [false_posi_num, true_nege_num]])
        # print("ob: {}".format(ob))

        chi2, p, dof, expected = chi2_contingency(ob)
        # print("chi2: {}, p: {}, dof: {}".format(chi2, p, dof))
        # print("Expected: {}".format(expected))
        return p

    """
    Get rules of relation (r_id)
    Parameters:
    ------------
    r_id, int, the id of relation
    
    Returns:
    ----------
    out: list, [[order,p_value,[r_id,r_id,..,r_id],[r_name,r_name,...,r_name]],[],...,[]]
    """

    def get_rules(self, r_id):

        print("Generate rules for : {}".format(r_id))

        rules_final = []
        if os.path.exists(self.rule_file):
            for line in open(self.rule_file, 'r', encoding="UTF-8").readlines():
                order, p_value, r_id_path, r_name_path = line.split("\t")
                rules_final.append(
                    [order, p_value, list(map(lambda x: int(x), r_id_path.split(" "))), r_name_path.split(" ")])
            return rules_final

        train_data = np.load(self.train_file)
        train_data = list(train_data[np.where(train_data[:, 2] == 1)])

        print("train data num : {}".format(len(train_data)))
        num_for_extract_rules = 10000
        if len(train_data) > num_for_extract_rules:
            train_data = np.array(train_data)
            train_data_idxes = np.random.choice(list(range(len(train_data))), num_for_extract_rules, False)
            train_data = train_data[np.array(train_data_idxes)]

        self.load_e_r_mapping()
        self.construct_kg()

        print("Max step: {}".format(self.max_step))
        res_r_id_path_list, searched_e_r_path = self.search_path(r_id, self.max_step, train_data)

        cnt = 0
        for r_id_path in res_r_id_path_list:
            # p_value = self.chi_square(r_id, r_id_path)
            p_value = 0
            if p_value < self.chi_thresh:
                rules_final.append([cnt, p_value, r_id_path, self.display_r_path(r_id_path)])
                cnt += 1

        with open(self.rule_file, 'w', encoding="UTF-8") as f:
            for order, p_value, r_id_path, r_name_path in rules_final:
                f.write("{}\t{}\t{}\t{}\n".format(order, p_value, " ".join(map(lambda x: str(x), r_id_path)),
                                                  " ".join(r_name_path)))
        return rules_final

    def get_o_list(self, s_id, r_id):
        if s_id not in self.kg:
            return []
        if r_id not in self.kg[s_id]:
            return []
        return self.kg[s_id][r_id]

    '''
    Check if a h t can pass the rule
    Parameters:
    -----------
    so: list, [subject,object]
    rule: list, [r_idx,r_idx,...]

    Returns:
    -----------
    out: boolean
    If ht passed rule, return Ture, False otherwises.
    '''

    def is_passed(self, so, rule):
        left_node = [so[0]]
        right_node = [so[-1]]
        left_step = 0
        right_step = 0
        while len(rule) - (left_step + right_step) > 0:
            temp_left = []
            temp_right = []
            if len(left_node) < len(right_node):
                left_step += 1
                r_idx = rule[left_step - 1]
                if r_idx not in self.r_id2name:
                    return False
                for e_idx in left_node:
                    for tail in self.get_o_list(e_idx, r_idx):
                        temp_left.append(tail)
                left_node = temp_left
            else:
                right_step += 1
                r_idx = rule[-right_step]
                if r_idx not in self.r_id2name:
                    return False
                inv_r_idx = self.convert_r(r_idx)
                for e_idx in right_node:
                    for tail in self.get_o_list(e_idx, inv_r_idx):
                        temp_right.append(tail)
                right_node = temp_right
        left_set = set()
        for e_idx in left_node:
            left_set.add(e_idx)
        for e_idx in right_node:
            if e_idx in left_set:
                return True
        return False

    """
    Get features for one pair of h and t
    Parameters:
    -----------
    rule_list: list, [[p1,p2,..,pn],,...]
    so: list, [[h,t],[h,t],[h,t],..]
    a list of s and o

    Returns:
    -----------
    out: list
    A list of features, every entry represents if a rule is passed.
    """

    def get_features(self, rule_list, so_list):
        train_x = []
        for ht_i, ht in enumerate(so_list):
            print("Feature: {}/{}, #Rules: {}, H: {}, T: {}".format(
                ht_i + 1, len(so_list), len(rule_list), self.e_id2name[ht[0]],
                self.e_id2name[ht[1]]))
            feature = []
            for rule in rule_list:
                feature.append(int(self.is_passed(ht, rule)))
            train_x.append(feature)
        return train_x

    def r_id_by_name(self, r_name):
        if r_name in self.r_name2id:
            return self.r_name2id[r_name]
        return None


if __name__ == "__main__":
    g = Graph(args)
    g.load_e_r_mapping()
    g.construct_kg()
    # g.split_train_eval_test(g.r_name2id['interact'])
    rules_final = g.get_rules(g.r_name2id['interact'])
