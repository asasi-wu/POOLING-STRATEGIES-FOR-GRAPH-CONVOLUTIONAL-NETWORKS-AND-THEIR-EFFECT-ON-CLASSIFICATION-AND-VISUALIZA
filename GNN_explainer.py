from math import sqrt
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, to_networkx
import torch.nn.functional as F
EPS = 1e-15


class GNNExplainer(torch.nn.Module):

    coeffs = {
       
        'node_feat_size': 0.01,
      
        'node_feat_ent': 0.5,
    }

    def __init__(self, model, epochs=100, lr=0.01):
        super(GNNExplainer, self).__init__()
        self.model = model
        self.epochs = epochs
        self.lr = lr
    

    def __set_masks__(self, x, edge_index, init="normal"):
        #(N, F), E = x.size(), edge_index.size(1)
        N,E=x.size(0),edge_index.size(1)
        std = 0.1
        #self.node_feat_mask = torch.nn.Parameter(torch.randn(F) * 0.1)
        self.node_mask = torch.nn.Parameter(torch.randn(N) * 0.01)
        #std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
        #self.edge_mask = torch.nn.Parameter(torch.randn(E) * std)
        """
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = True
                module.__edge_mask__ = self.edge_mask
        """
                
                
    def __clear_masks__(self):
        self.node_masks = None
        #self.edge_mask = None


    def __loss__(self, log_logits, pred_label):
        #loss = -log_logits[0][pred_label]
        loss=F.nll_loss(log_logits,pred_label)
        #loss=F.nll_loss(log_logits,pred_label,reduction='sum').item()
        """
        m = self.edge_mask.sigmoid()
        loss = loss + self.coeffs['edge_size'] * m.sum()
        ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
        loss = loss + self.coeffs['edge_ent'] * ent.mean()
        """
        m = self.node_mask.sigmoid()
        #loss = loss + self.coeffs['node_feat_size'] * m.sum()
        
        ent = -m * torch.log(m + EPS)- (1 - m) * torch.log(1 - m + EPS)
        #print(ent)
        loss = loss + self.coeffs['node_feat_ent'] * ent.mean()
        return loss

    def explain_graph(self, x, edge_index, **kwargs):

        self.model.eval()
        self.__clear_masks__()

        # num_edges = edge_index.size(1)

        # Get the initial prediction.
        with torch.no_grad():
            log_logits = self.model(x, edge_index, **kwargs)
            pred_label = log_logits.argmax(dim=-1)

        self.__set_masks__(x, edge_index)
        self.to(x.device)

        optimizer = torch.optim.Adam([self.node_mask],lr=self.lr)


        for epoch in range(1, self.epochs + 1):
            optimizer.zero_grad()
    
            h = x * self.node_mask.reshape((self.node_mask.shape[0],1)).sigmoid()
            log_logits = self.model(h, edge_index, **kwargs)
            loss = self.__loss__(log_logits, pred_label)
            loss.backward()
            optimizer.step()
            
            

        node_mask = self.node_mask.detach().sigmoid()
        
        final_mask=node_mask.cpu().data.numpy()
        final_mask=final_mask.argsort()[::-1]
        
        #edge_mask = self.edge_mask.new_zeros(num_edges)
        #edge_mask[hard_edge_mask] = self.edge_mask.detach().sigmoid()

        self.__clear_masks__()
        return final_mask
