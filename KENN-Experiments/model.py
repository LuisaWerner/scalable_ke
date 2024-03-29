""" Base Neural Network and Knowledge Enhanced Models """

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv

from kenn.parsers import *


def get_model(data, args):
    """ instantiates the model specified in args """

    # Base neural network
    if not args.model.startswith('KENN'):
        if args.model == 'MLP':
            _class = MLP
        elif args.model == 'GCN':
            _class = GCN
        elif args.model == 'SAGE':
            _class = SAGE
        else:
            _class = 'MLP'
            print('Unknown model, set to MLP')

        model = _class(in_channels=data.num_features,
                       out_channels=data.num_classes,
                       hidden_channels=args.hidden_channels,
                       num_layers=args.num_layers,
                       dropout=args.dropout)

    # kenn-sub network
    elif args.model.startswith('KENN'):
        if args.model == 'KENN_MLP':
            _class = KENN_MLP
        elif args.model == 'KENN_GCN':
            _class = KENN_GCN
        elif args.model == 'KENN_SAGE':
            _class = KENN_SAGE
        else:
            _class = 'MLP'
            print('Unknown model, set to MLP')

        model = _class(knowledge_file='knowledge_base',
                       in_channels=data.num_features,
                       out_channels=data.num_classes,
                       hidden_channels=args.hidden_channels,
                       num_layers=args.num_layers,
                       num_kenn_layers=args.num_kenn_layers,
                       dropout=args.dropout)
    else:
        print(f'Value Error: {args.model} does not exist. Choose a model in the list: GCN, SAGE, MLP, KENN_GCN, '
              f'KENN_SAGE, KENN_MLP')
        model = None

    return model


class GCN(torch.nn.Module):
    """
    GCN module baseline given by OGB
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super().__init__()
        self.name = 'GCN'
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index, relations, edge_weight=None):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, edge_weight)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class SAGE(torch.nn.Module):
    """ Implementation of GraphSAGE - ogb baseline  """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super().__init__()
        self.name = 'SAGE'
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index, relations, edge_weight=None):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, edge_weight)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class MLP(torch.nn.Module):
    """ MLP baseline for OGB """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super().__init__()
        self.name = 'MLP'
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index, relations=None, edge_weight=None):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x


class Standard(torch.nn.Module):
    """ Original Base Neural Network used in KENN paper """

    # in_channels =
    # out_channels = 6
    # hidden_channels = 50

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super().__init__()
        self.lin_layers = torch.nn.ModuleList()
        self.lin_layers.append(torch.nn.Linear(in_channels, 50))
        self.lin_layers.append(torch.nn.Linear(50, 50))
        self.lin_layers.append(torch.nn.Linear(50, 50))
        self.lin_layers.append(torch.nn.Linear(50, out_channels))
        self.dropout = 0.5

    def reset_parameters(self):
        for layer in self.lin_layers:
            layer.reset_parameters()

    def forward(self, x, edge_index=None, relations=None, edge_weight=None):
        for i, lin in enumerate(self.lin_layers[:-1]):
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin_layers[-1](x)
        return x


class KENN_GCN(GCN):
    """ relational KENN with GCN as base NN"""

    def __init__(self, knowledge_file, hidden_channels, in_channels, out_channels, num_layers, num_kenn_layers,
                 dropout):
        super().__init__(in_channels=in_channels, out_channels=out_channels,
                         hidden_channels=hidden_channels,
                         num_layers=num_layers, dropout=dropout)
        self.name = str('KENN_' + self.name)
        self.knowledge_file = knowledge_file
        self.kenn_layers = torch.nn.ModuleList()

        for _ in range(num_kenn_layers):
            self.kenn_layers.append(relational_parser(knowledge_file=knowledge_file))

    def reset_parameters(self):
        super().reset_parameters()
        for layer in self.kenn_layers:
            layer.reset_parameters()

    def forward(self, x, edge_index, relations, edge_weight=None):
        z = super().forward(x, edge_index, relations, edge_weight)

        # call kenn-sub layers
        for layer in self.kenn_layers:
            z, _ = layer(unary=z, edge_index=edge_index, edge_weight=edge_weight, binary=relations)

        return z


class KENN_MLP(MLP):
    """ KENN with MLP (from ogb) as base NN"""

    def __init__(self, knowledge_file, hidden_channels, in_channels, out_channels, num_layers, num_kenn_layers,
                 dropout):
        super().__init__(in_channels=in_channels, out_channels=out_channels,
                         hidden_channels=hidden_channels,
                         num_layers=num_layers, dropout=dropout)
        self.name = str('KENN_' + self.name)
        self.knowledge_file = knowledge_file
        self.kenn_layers = torch.nn.ModuleList()

        for _ in range(num_kenn_layers):
            self.kenn_layers.append(relational_parser(knowledge_file=knowledge_file))

    def reset_parameters(self):
        super().reset_parameters()
        for layer in self.kenn_layers:
            layer.reset_parameters()

    def forward(self, x, edge_index, relations=None, edge_weight=None):
        z = super().forward(x, edge_index, relations, edge_weight)

        # call kenn-sub layers
        for layer in self.kenn_layers:
            z, _ = layer(unary=z, edge_index=edge_index, edge_weight=edge_weight, binary=relations)

        return z


class KENN_SAGE(SAGE):
    """ kenn-sub with GraphSage (from ogb) as base NN"""

    def __init__(self, knowledge_file, hidden_channels, in_channels, out_channels, num_layers, num_kenn_layers,
                 dropout):
        super().__init__(in_channels=in_channels, out_channels=out_channels,
                         hidden_channels=hidden_channels,
                         num_layers=num_layers, dropout=dropout)
        self.name = str('KENN_' + self.name)
        self.knowledge_file = knowledge_file
        self.kenn_layers = torch.nn.ModuleList()

        for _ in range(num_kenn_layers):
            self.kenn_layers.append(relational_parser(knowledge_file=knowledge_file))

    def reset_parameters(self):
        super().reset_parameters()
        for layer in self.kenn_layers:
            layer.reset_parameters()

    def forward(self, x, edge_index, relations, edge_weight=None):
        z = super().forward(x, edge_index, relations, edge_weight)

        # call kenn-sub layers
        for layer in self.kenn_layers:
            z, _ = layer(unary=z, edge_index=edge_index, binary=relations)

        return z
