import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.conv import MessagePassing
from typing import Optional, Tuple
from torch_geometric.typing import Adj, OptTensor
from torch import Tensor
from torch_sparse import SparseTensor, matmul
from torch.nn import Linear
from torch import nn
from torch_geometric.nn import global_max_pool
from torch_geometric.nn import aggr
import torch.nn.functional as F
from torch_geometric.nn import APPNP, MLP, GCNConv, GINConv, SAGEConv, GraphConv, TransformerConv, ChebConv, GATConv, SGConv, GeneralConv
from torch.nn import Conv1d, MaxPool1d, ModuleList
import random
import math
softmax = torch.nn.LogSoftmax(dim=1)

class GNNs(torch.nn.Module):
    def __init__(self,args, train_dataset, hidden_channels, num_layers, GNN, k=0.6):
        super().__init__()
        if k < 1:  # Transform percentile to number.
            num_nodes = sorted([data.num_nodes for data in train_dataset])
            k = num_nodes[int(math.ceil(k * len(num_nodes))) - 1]
            k = max(10, k)
        self.k = int(k)
        self.sort_aggr = aggr.SortAggregation(self.k)
        self.convs = ModuleList()
        if args.model=="ChebConv":
            self.convs.append(GNN(train_dataset.num_features, hidden_channels,K=5))
#         self.convs.append(GNN(1433, hidden_channels))
            for i in range(0, num_layers - 1):
                self.convs.append(GNN(hidden_channels, hidden_channels,K=5))
            self.convs.append(GNN(hidden_channels, 1,K=5))
        else:
            self.convs.append(GNN(train_dataset.num_features, hidden_channels))
    #         self.convs.append(GNN(1433, hidden_channels))
            for i in range(0, num_layers - 1):
                self.convs.append(GNN(hidden_channels, hidden_channels))
            self.convs.append(GNN(hidden_channels, 1))
#         self.lin = Linear(hidden_channels*(num_layers+1), dataset.num_classes)
        conv1d_channels = [16, 32]
        total_latent_dim = hidden_channels * num_layers + 1
        conv1d_kws = [total_latent_dim, 5]
        self.conv1 = Conv1d(1, conv1d_channels[0], conv1d_kws[0],
                            conv1d_kws[0])
        self.maxpool1d = MaxPool1d(2, 2)
        self.conv2 = Conv1d(conv1d_channels[0], conv1d_channels[1],
                            conv1d_kws[1], 1)
        dense_dim = int((self.k - 2) / 2 + 1)
        dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]
        self.mlp = MLP([dense_dim, 32, args.num_classes], dropout=0.5, batch_norm=False)
    def reset_parameters(self):
        # for conv in self.convs:
        #     conv.reset_parameter()
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.mlp.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        xs = [x]        
        for conv in self.convs:
            xs += [conv(xs[-1], edge_index).tanh()]
#             x = conv(x, edge_index).tanh()
        x = torch.cat(xs[1:], dim=-1)

        
        x = self.sort_aggr(x,batch)
        x = x.unsqueeze(1)  # [num_graphs, 1, k * hidden]
        x = self.conv1(x).relu()
        x = self.maxpool1d(x)
        x = self.conv2(x).relu()
        x = x.view(x.size(0), -1)  # [num_graphs, dense_dim]
        x = self.mlp(x)
        return x


class NN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        x = self.fc3(x)
        
        return x
    
class Attention(nn.Module):
    def __init__(self, input_dim1, input_dim2):
        super(Attention, self).__init__()
        self.alpha1 = nn.Parameter(torch.zeros(input_dim1))
        self.alpha2 = nn.Parameter(torch.zeros(input_dim2))

    def forward(self, x1, x2):
        alpha1 = torch.sigmoid(self.alpha1)
        alpha2 = torch.sigmoid(self.alpha2)
        x1_weighted = alpha1 * x1
        x2_weighted = alpha2 * x2
        return torch.cat((x1_weighted, x2_weighted), dim=-1)

class ResidualGNNs(torch.nn.Module):
    def __init__(self,args, train_dataset, hidden_channels,hidden, num_layers, GNN, k=0.6):
        super().__init__()
        self.convs = ModuleList()
        self.aggr = aggr.MeanAggregation()
        self.hidden_channels = hidden_channels
        num_features = train_dataset.num_features
        if args.model=="ChebConv":
            if num_layers>0:
                self.convs.append(GNN(num_features, hidden_channels,K=5))
                for i in range(0, num_layers - 1):
                    self.convs.append(GNN(hidden_channels, hidden_channels,K=5))
        else:
            if num_layers>0:
                self.convs.append(GNN(num_features, hidden_channels))
                for i in range(0, num_layers - 1):
                    self.convs.append(GNN(hidden_channels, hidden_channels))
        
        input_dim1 = int(((num_features * num_features)/2)- (num_features/2)+(hidden_channels*num_layers))
        input_dim = int(((num_features * num_features)/2)- (num_features/2))
        self.bn = nn.BatchNorm1d(input_dim)
        self.bnh = nn.BatchNorm1d(hidden_channels*num_layers)
        # self.attention = Attention(input_dim1, hidden_channels)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim1, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden, hidden//2),
            nn.BatchNorm1d(hidden//2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden//2, hidden//2),
            nn.BatchNorm1d(hidden//2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear((hidden//2), args.num_classes),
        )
        
    
    
    # def reset_parameters(self):
    #     # for conv in self.convs:
    #     #     conv.reset_parameter()
    #     self.conv1.reset_parameters()
    #     self.conv2.reset_parameters()
    #     self.mlp.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # for conv in self.convs:
        #     x = conv(x, edge_index).relu()
        
        xs = [x]        
        for conv in self.convs:
            xs += [conv(xs[-1], edge_index).tanh()]

        # xx = data.x.reshape(data.num_graphs, data.x.shape[1],-1)
        # h.append(torch.stack([t.triu().flatten()[t.triu().flatten().nonzero(as_tuple=True)] for t in xx]))
        # x = self.aggr(x,batch)
        # h.append(x)
        # xx = torch.stack([t.triu().flatten()[t.triu().flatten().nonzero(as_tuple=True)] for t in xx])
        # x = self.aggr(x,batch)
        # xx = self.bn(xx)
        # x = self.bnh(x)
        # x = self.attention(xx,x)
        h = []
        for i, xx in enumerate(xs):
            if i== 0:
                xx = xx.reshape(data.num_graphs, x.shape[1],-1)
                x = torch.stack([t.triu().flatten()[t.triu().flatten().nonzero(as_tuple=True)] for t in xx])
                x = self.bn(x)
            else:
                # xx = xx.reshape(data.num_graphs, x.shape[1],-1)
                xx = self.aggr(xx,batch)
                # h.append(torch.stack([t.flatten() for t in xx]))
                h.append(xx)

        h = torch.cat(h,dim=1)
        h = self.bnh(h)
        # x = torch.stack(h, dim=0)
        x = torch.cat((x,h),dim=1)
        x = self.mlp(x)
        return x

class AirGC(torch.nn.Module):
    def __init__(self, dataset, args):
        super(AirGC, self).__init__()

        self.prop = AdaptiveMessagePassing(K=args.K, alpha=args.alpha, mode=args.model, args=args)

        input_dim = dataset.num_features * dataset.num_features
        self.bn = nn.BatchNorm1d(input_dim)

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, args.hidden),
            nn.BatchNorm1d(args.hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(args.hidden, args.hidden//2),
            nn.BatchNorm1d(args.hidden//2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(args.hidden//2, args.hidden//2),
            nn.BatchNorm1d(args.hidden//2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear((args.hidden//2), dataset.num_classes)
        )

    def reset_parameters(self):
        self.prop.reset_parameters()
        self.mlp.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.adj_t
        x = self.prop(x, edge_index)
        x = x.reshape(data.num_graphs, x.shape[1], -1)
        x = x.flatten(start_dim=1, end_dim=2)
        x = self.bn(x)
        x = self.mlp(x)
        return F.log_softmax(x, dim=1)


### UNCHANGED FROM AIRGNN REPO https://github.com/lxiaorui/AirGNN/blob/main/model.py ####
class AdaptiveMessagePassing(MessagePassing):
    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self,
                 K: int,
                 alpha: float,
                 dropout: float = 0.,
                 cached: bool = False,
                 add_self_loops: bool = True,
                 normalize: bool = True,
                 mode: str = None,
                 node_num: int = None,
                 args=None,
                 **kwargs):

        super(AdaptiveMessagePassing, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.alpha = alpha
        self.mode = mode
        self.dropout = dropout
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self._cached_edge_index = None
        self.node_num = node_num
        self.args = args
        self._cached_adj_t = None

    def reset_parameters(self):
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, edge_index: Adj, edge_weight: OptTensor = None, mode=None) -> Tensor:
        if self.normalize:
            if isinstance(edge_index, Tensor):
                raise ValueError('Only support SparseTensor now')

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim), False,
                        add_self_loops=self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        if mode == None: mode = self.mode

        if self.K <= 0:
            return x
        hh = x

        if mode == 'MLP':
            return x

        elif mode == 'APPNP':
            x = self.appnp_forward(x=x, hh=hh, edge_index=edge_index, K=self.K, alpha=self.alpha)

        elif mode in ['AirGC', 'AirGNN']:
            x = self.amp_forward(x=x, hh=hh, edge_index=edge_index, K=self.K)
        else:
            raise ValueError('wrong propagate mode')
        return x

    def appnp_forward(self, x, hh, edge_index, K, alpha):
        for k in range(K):
            x = self.propagate(edge_index, x=x, edge_weight=None, size=None)
            x = x * (1 - alpha)
            x += alpha * hh
        return x

    def amp_forward(self, x, hh, K, edge_index):
        lambda_amp = self.args.lambda_amp
        gamma = 1 / (2 * (1 - lambda_amp))  ## or simply gamma = 1

        for k in range(K):
            y = x - gamma * 2 * (1 - lambda_amp) * self.compute_LX(x=x, edge_index=edge_index)  # Equation (9)
            x = hh + self.proximal_L21(x=y - hh, lambda_=gamma * lambda_amp) # Equation (11) and (12)
        return x

    def proximal_L21(self, x: Tensor, lambda_):
        row_norm = torch.norm(x, p=2, dim=1)
        score = torch.clamp(row_norm - lambda_, min=0)
        index = torch.where(row_norm > 0)             #  Deal with the case when the row_norm is 0
        score[index] = score[index] / row_norm[index] # score is the adaptive score in Equation (14)
        return score.unsqueeze(1) * x

    def compute_LX(self, x, edge_index, edge_weight=None):
        x = x - self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)
        return x

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}(K={}, alpha={}, mode={}, dropout={}, lambda_amp={})'.format(self.__class__.__name__, self.K,
                                                               self.alpha, self.mode, self.dropout,
                                                               self.args.lambda_amp)