from torch_geometric.nn import GCNConv
from torch_geometric.nn.pool.topk_pool import filter_adj
from torch.nn import Parameter
import torch
import numpy
from torch_geometric.utils import dense_to_sparse,from_scipy_sparse_matrix
import torch.nn.functional as F
from collections import namedtuple
from torch_scatter import scatter_add
from torch_sparse import coalesce
from torch_geometric.utils import softmax
from collections import namedtuple
from torch_scatter import scatter_add
from torch_sparse import coalesce
from torch_geometric.utils import softmax

def topk(x, ratio, batch, min_score=None, tol=1e-7):

    num_nodes = scatter_add(batch.new_ones(x.size(0)), batch, dim=0)
    batch_size, max_num_nodes = num_nodes.size(0), num_nodes.max().item()

    cum_num_nodes = torch.cat([num_nodes.new_zeros(1),num_nodes.cumsum(dim=0)[:-1]], dim=0)

    index = torch.arange(batch.size(0), dtype=torch.long, device=x.device)
    index = (index - cum_num_nodes[batch]) + (batch * max_num_nodes)

    dense_x = x.new_full((batch_size * max_num_nodes, ),torch.finfo(x.dtype).min)
    dense_x[index] = x
    dense_x = dense_x.view(batch_size, max_num_nodes)

    _, perm = dense_x.sort(dim=-1, descending=True)

    perm = perm + cum_num_nodes.view(-1, 1)
    perm = perm.view(-1)

    k = (ratio * num_nodes.to(torch.float)).ceil().to(torch.long)
    mask = [
        torch.arange(k[i], dtype=torch.long, device=x.device) +
        i * max_num_nodes for i in range(batch_size)
        ]
    mask = torch.cat(mask, dim=0)

    perm = perm[mask]

    return perm


def to_dense_adj(edge_index, batch=None, edge_attr=None):
    if batch is None:
        batch = edge_index.new_zeros(edge_index.max().item() + 1)
    batch_size = batch[-1].item() + 1  
    one = batch.new_ones(batch.size(0))
    num_nodes = scatter_add(one, batch, dim=0, dim_size=batch_size)
    cum_nodes = torch.cat([batch.new_zeros(1), num_nodes.cumsum(dim=0)])
    max_num_nodes = num_nodes.max().item()
    size = [ max_num_nodes, max_num_nodes]
    #size = [batch_size, max_num_nodes, max_num_nodes]
    size = size 
    dtype = torch.float
    adj = torch.zeros(size, dtype=dtype, device=edge_index.device)  
    edge_index_0 = batch[edge_index[0]].view(1, -1)
    edge_index_1 = edge_index[0] - cum_nodes[batch][edge_index[0]]
    edge_index_2 = edge_index[1] - cum_nodes[batch][edge_index[1]]
    if edge_attr is None:
        adj[ edge_index_1, edge_index_2] = 1
        #adj[edge_index_0, edge_index_1, edge_index_2] = 1
    else:
        adj[edge_index_0, edge_index_1, edge_index_2] = edge_attr
    return adj

def generate_mask(s_l_app,batch,s_l):
        batch_size = batch[-1].item() + 1
        one = batch.new_ones(batch.size(0))
    
        num_each_nodes = scatter_add(one, batch, dim=0, dim_size=batch_size)

        number_nodes = torch.cat([batch.new_zeros(1), num_each_nodes.cumsum(dim=0)])
        for i in range(batch_size):
            s_l_app[number_nodes[i]:number_nodes[i+1],s_l.shape[1]*i:((i+1)*s_l.shape[1])]=1
        mask=s_l_app
        return mask



class DiffPool(torch.nn.Module):
    
    def __init__(self, in_channels,ratio=0.8,Conv=GCNConv,non_linearity=torch.sigmoid,is_final=False, link_pred=False, nhid=128):
        super(DiffPool, self).__init__()
        self.nfeat=in_channels
        self.nnext1=64
        self.nnext2=32
        self.nnext3=16
        nhid=in_channels
        self.link_pred = link_pred
        self.is_final = is_final
        self.embed = Conv(self.nfeat, nhid)
        self.assign_mat1 = Conv(self.nfeat, self.nnext1)
        self.assign_mat2 = Conv(self.nfeat, self.nnext2)
        self.assign_mat3 = Conv(self.nfeat, self.nnext3)
        self.link_pred_loss = 0
        self.entropy_loss = 0

    def forward(self, x, edge_index, edge_attr=None, batch=None,layer=None):
        batch_size = batch[-1].item() + 1
        
        adj=to_dense_adj(edge_index)
        if adj.shape[0]!=x.shape[0]:
            print("something wrong")
            x=x[:adj.shape[0],:]
        #print("adj.size()")
        #print(adj.size())
        z_l = self.embed(x, edge_index) 
        #z_l=x
        if layer=="layer2":
            s_l = F.softmax(self.assign_mat2(x, edge_index), dim=-1)
            nnext=32
        elif layer=="layer3":
            s_l = F.softmax(self.assign_mat3(x, edge_index), dim=-1)
            nnext=16
        else:
            s_l = F.softmax(self.assign_mat1(x, edge_index), dim=-1)
            nnext=64
        s_l_new=s_l
        #print("s_l_new.size()")
        #print(s_l_new.size())
        for i in range(batch_size-1):     
            s_l_new = torch.cat([s_l_new, s_l],1)
        #s_l_app=torch.zeros([s_l.shape[0],int(s_l.shape[1]*128)], dtype=torch.float, device='cuda:0')
        mask=generate_mask(torch.zeros([s_l.shape[0],int(s_l.shape[1]*batch_size)], dtype=torch.float, device='cuda:0'),batch,s_l)
        s_l_app=s_l_new*mask 
        xnext = torch.matmul(s_l_app.transpose(-1, -2),z_l)
        anext = (s_l_app.t()).matmul(adj).matmul(s_l_app)
        edge_index, _=dense_to_sparse(anext)
        assert edge_index.shape[0] == 2
        #batch = torch.zeros(xnext.shape[0], dtype=torch.int, device='cuda:0')
        batch=batch.new_zeros(xnext.shape[0])
        for i in range (batch_size):
            batch[i*nnext:((i+1)*nnext)]=i  
        temp1=0
        temp2=0
        return xnext, edge_index,temp1,batch,temp2




class EdgePooling(torch.nn.Module):


    def __init__(self, in_channels, ratio=0.8,Conv=GCNConv,non_linearity=torch.sigmoid):
        super(EdgePooling, self).__init__()
        self.in_channels = in_channels
        self.add_to_edge_score = 0.5
        self.dropout = 0
        self.lin = torch.nn.Linear(2 * in_channels, 1)
    def forward(self, x, edge_index, edge_attr=None, batch=None,layer=None):

        e = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=-1)
        e = self.lin(e).view(-1)
        #e = F.dropout(e, p=self.dropout, training=self.training)
        e = torch.tanh(e)
        e = e + self.add_to_edge_score

        x, edge_index, batch = self.__merge_edges__(
            x, edge_index, batch, e)
        temp1=0
        temp2=0
        return x, edge_index,temp1, batch,temp2


    def __merge_edges__(self, x, edge_index, batch, edge_score):
        nodes_remaining = set(range(x.size(0)))

        cluster = torch.empty_like(batch, device=torch.device('cpu'))
        edge_argsort = torch.argsort(edge_score, descending=True)

        # Iterate through all edges, selecting it if it is not incident to
        # another already chosen edge.
        i = 0
        new_edge_indices = []
        edge_index_cpu = edge_index.cpu()
        for edge_idx in edge_argsort.tolist():
            source = edge_index_cpu[0, edge_idx].item()
            if source not in nodes_remaining:
                continue
            target = edge_index_cpu[1, edge_idx].item()
            if target not in nodes_remaining:
                continue
            new_edge_indices.append(edge_idx)
            cluster[source] = i
            nodes_remaining.remove(source)
            if source != target:
                cluster[target] = i
                nodes_remaining.remove(target)
            i += 1

        # The remaining nodes are simply kept.
        for node_idx in nodes_remaining:
            cluster[node_idx] = i
            i += 1
        cluster = cluster.to(x.device)

        # We compute the new features as an addition of the old ones.
        new_x = scatter_add(x, cluster, dim=0, dim_size=i)
        new_edge_score = edge_score[new_edge_indices]
        if len(nodes_remaining) > 0:
            remaining_score = x.new_ones((new_x.size(0) - len(new_edge_indices), ))
            new_edge_score = torch.cat([new_edge_score, remaining_score])
        new_x = new_x * new_edge_score.view(-1, 1)

        N = new_x.size(0)
        new_edge_index, _ = coalesce(cluster[edge_index], None, N, N)

        new_batch = x.new_empty(new_x.size(0), dtype=torch.long)
        new_batch = new_batch.scatter_(0, cluster, batch)
    
        return new_x, new_edge_index,new_batch

    

class gPool(torch.nn.Module):
    def __init__(self,in_channels,ratio=0.8,Conv=GCNConv,non_linearity=torch.sigmoid):
        super(gPool,self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        
        self.non_linearity = non_linearity
        self.coefficient = Parameter(torch.FloatTensor(in_channels,1 ).zero_())
    def forward(self, x, edge_index, edge_attr=None, batch=None,layer=None):
    
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        score = torch.mm(x,self.coefficient).squeeze()
        #print("score:{}\t score size:{}".format(score,score.size()))
        
        perm = topk(score, self.ratio, batch)
        x = x[perm] * self.non_linearity(score[perm]).view(-1, 1)
        batch = batch[perm]
        edge_index, edge_attr = filter_adj(
            edge_index, edge_attr, perm, num_nodes=score.size(0))

        return x, edge_index, edge_attr, batch, perm
    
    
    
class SAGPool(torch.nn.Module):
    def __init__(self,in_channels,ratio=0.8,Conv=GCNConv,non_linearity=torch.sigmoid):
        super(SAGPool,self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.score_layer = Conv(in_channels,1)
        
        self.non_linearity = non_linearity
    def forward(self, x, edge_index, edge_attr=None, batch=None,layer=None):

        
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        score = self.score_layer(x,edge_index).squeeze()
        perm = topk(score, self.ratio, batch)
        x = x[perm]* self.non_linearity(score[perm]).view(-1, 1)
        batch = batch[perm]
        edge_index, edge_attr = filter_adj(
            edge_index, edge_attr, perm, num_nodes=score.size(0))

        return x, edge_index, edge_attr, batch, perm



