from abc import abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F

LAYER_IDS = {}


def get_layer_id(layer_name=''):
    if layer_name not in LAYER_IDS:
        LAYER_IDS[layer_name] = 0
        return 0
    else:
        LAYER_IDS[layer_name] += 1
        return LAYER_IDS[layer_name]


class Aggregator(object):
    def __init__(self, batch_size, dim, dropout, act, name):
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_id(layer))
        self.name = name
        self.dropout = dropout
        self.act = act
        self.batch_size = batch_size
        self.dim = dim

    def __call__(self, self_vectors, neighbor_vectors):
        outputs = self._call(self_vectors, neighbor_vectors)
        return outputs

    @abstractmethod
    def _call(self, self_vectors, neighbor_vectors):
        # dimension:
        # self_vectors: [batch_size, -1, dim] ([batch_size, -1] for LabelAggregator)
        # neighbor_vectors: [batch_size, -1, n_neighbor, dim] ([batch_size, -1, n_neighbor] for LabelAggregator)
        pass

    def _mix_neighbor_vectors(self, neighbor_vectors):
        # neighbor_vectors, [batch_size, -1, neighbor_size, dim]
        # [batch_size, -1, dim]
        neighbors_aggregated = torch.mean(neighbor_vectors, dim=2)

        return neighbors_aggregated


class SumAggregator(Aggregator):
    def __init__(self, batch_size, dim, dropout=0., act=F.relu, name=None):
        super(SumAggregator, self).__init__(batch_size, dim, dropout, act, name)

        self.l1 = nn.Linear(self.dim, self.dim)

    # self_vectors, [batch_size,-1,dim]
    # neighbor_vectors, [batch_size,-1,neighbour_size,dim]
    def _call(self, self_vectors, neighbor_vectors):
        # [batch_size, -1, dim]
        neighbors_agg = self._mix_neighbor_vectors(neighbor_vectors)

        # [-1, dim]
        output = (self_vectors + neighbors_agg).view([-1, self.dim])
        output = F.dropout(output, p=self.dropout)
        output = self.l1(output)

        # [batch_size, -1, dim]
        output = output.view([self.batch_size, -1, self.dim])

        return self.act(output)
