import torch
import torch.nn as nn
import numpy as np
import os

from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, roc_curve

from ALogger import ALogger
from Args import args
from Graph import Graph
from Sundries import get_x_percent_of_ndarray

device = "cuda" if torch.cuda.is_available() else "cpu"


class PRA(nn.Module):
    def __init__(self, feature_size, lr, l2):
        super(PRA, self).__init__()
        self.logger = ALogger("PRA", True).getLogger()
        self.feature_size = feature_size
        self.lr = lr
        self.l2 = l2

        self._build_model()

    def _build_model(self):
        self.layer1 = nn.Linear(self.feature_size, 1)

    # @torchsnooper.snoop()
    def forward(self, input_x):
        return self.layer1(torch.Tensor(input_x).to(device))

    # @torchsnooper.snoop()
    def update(self, train_x, train_y):
        output = self.forward(train_x)
        output = torch.sigmoid(output)
        output = output.view(-1)

        criterion = nn.BCELoss().to(device)
        loss = criterion(output, torch.Tensor(train_y).to(device))
        loss.backward()

        return loss.cpu().item() / len(train_x)

    def save_model(self, file_path):
        self.logger.info("Save model to {}.".format(file_path))
        torch.save(self.state_dict(), file_path)

    def load_model(self, file_path):
        self.logger.info("Load model from {}.".format(file_path))
        self.load_state_dict(torch.load(file_path))

    def save_rule_weight(self, rule_weight_path):
        layer_weight = self.layer1.weight.data.cpu().numpy()
        np.save(rule_weight_path, layer_weight)


# @torchsnooper.snoop()
def train_pra_model(pra_args, pra_model, train_data, eval_data, test_data):
    lr = pra_args.pra_lr
    l2_weight = pra_args.pra_l2_weight
    batch_size = pra_args.pra_batch_size
    n_epochs = pra_args.pra_n_epochs

    optimizer = torch.optim.Adam(pra_model.parameters(), lr=lr, weight_decay=l2_weight)

    eval_auc_list = []
    eval_f_list = []

    train_cnt = 0

    for epoch_i in range(n_epochs):
        start = 0
        np.random.shuffle(train_data)
        train_x = train_data[:, :-1]
        train_y = train_data[:, -1]

        running_loss = 0
        while start + batch_size < train_data.shape[0]:
            optimizer.zero_grad()
            loss = pra_model.update(train_x[start:start + batch_size], train_y[start:start + batch_size])
            start += batch_size
            running_loss += loss
            optimizer.step()

        running_loss /= start - batch_size

        if (epoch_i + 1) % 5 != 0:
            continue

        print("Epoch: {}/{}, Loss: {}"
              .format(epoch_i, n_epochs, running_loss))

        train_auc, train_prec, train_reca, train_f = eval_prec_reca_f(pra_model, train_data)
        eval_auc, eval_prec, eval_reca, eval_f = eval_prec_reca_f(pra_model, eval_data)
        test_auc, test_prec, test_reca, test_f = eval_prec_reca_f(pra_model, test_data)

        eval_auc_list.append(eval_auc)
        eval_f_list.append(eval_f)

        if len(eval_auc_list) == 1 or eval_auc_list[-1] > eval_auc_list[-2]:
            train_cnt = 0
            pra_model.save_model(args.pra_model_file_path)

        print('epoch %d    train auc: %.4f  f1: %.4f prec: %.4f reca: %.4f  '
              'eval auc: %.4f  f1: %.4f prec: %.4f reca: %.4f    '
              'test auc: %.4f  f1: %.4f prec: %.4f reca: %.4f'
              % (epoch_i, train_auc, train_f, train_prec, train_reca,
                 eval_auc, eval_f, eval_prec, eval_reca,
                 test_auc, test_f, test_prec, test_reca))

        if len(eval_auc_list) > 1 and eval_auc_list[-1] <= eval_auc_list[-2]:
            train_cnt += 1
            if train_cnt >= 3:
                break


def eval_prec_reca_f(pra_model, data):
    data_x = data[:, :-1]
    data_y = data[:, -1]

    scores = pra_model.forward(data_x).cpu().detach().numpy()
    scores[scores >= 0.5] = 1
    scores[scores < 0.5] = 0
    precision = precision_score(y_true=data_y, y_pred=scores)
    recall = recall_score(y_true=data_y, y_pred=scores)
    f1 = f1_score(y_true=data_y, y_pred=scores)

    auc = roc_auc_score(y_true=data_y, y_score=scores)

    return auc, precision, recall, f1


def generate_path_feature(rule_id_list, g, data_x_y, feature_data_path):
    if os.path.exists(feature_data_path):
        return np.load(feature_data_path)

    data_x = data_x_y[:, :-1]
    data_y = data_x_y[:, -1]
    data_y = data_y.reshape((-1, 1))
    rule_feature = g.get_features(rule_id_list, data_x)
    rule_feature_label = np.hstack((rule_feature, data_y))
    np.save(feature_data_path, rule_feature_label)
    return rule_feature_label


if __name__ == "__main__":

    train_model = True

    g = Graph(args)
    g.load_e_r_mapping()
    g.construct_kg()
    # g.construct_adj_e_id()

    rule_list = g.get_rules(g.r_name2id['interact'])
    rule_list.sort(key=lambda x: x[0])
    rule_id_list = [rule for _, _, rule, _ in rule_list]

    train_data = np.load(args.train_file)
    eval_data = np.load(args.eval_file)
    test_data = np.load(args.test_file)

    data_percent = 1
    train_data = get_x_percent_of_ndarray(data_percent, train_data)
    print("Train data num: {}".format(len(train_data)))

    eval_data = get_x_percent_of_ndarray(data_percent, eval_data)
    print("Eval data num: {}".format(len(eval_data)))

    test_data = get_x_percent_of_ndarray(data_percent, test_data)
    print("Test data num: {}".format(len(test_data)))

    train_feature_data = generate_path_feature(rule_id_list, g, train_data, args.train_feature_file)
    eval_feature_data = generate_path_feature(rule_id_list, g, eval_data, args.eval_feature_file)
    test_feature_data = generate_path_feature(rule_id_list, g, test_data, args.test_feature_file)

    pra_model = PRA(train_feature_data[:, 0:-1].shape[1], args.pra_lr, args.pra_l2_weight)

    pra_model.to(device)
    if train_model is True:
        train_pra_model(args, pra_model, train_feature_data, eval_feature_data, test_feature_data)
    else:
        pra_model.load_model(args.pra_model_file_path)

    pra_model.save_rule_weight(args.rule_weight_file_path)
