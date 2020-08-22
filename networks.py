import torch
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GraphConv, TopKPooling
from torch_geometric.nn import global_mean_pool as mean_pool, global_max_pool as max_pool
import torch.nn.functional as F
from Poolayer import SAGPool,gPool,EdgePooling,DiffPool


class Net(torch.nn.Module):
    def __init__(self,args):
        super(Net, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.hidden = args.hidden_size
        self.num_classes = args.num_classes
        self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout_ratio
        self.pool_method = args.pool_method
        
        if args.two_conv==True:
            self.conv1 = GCNConv(self.num_features, 128)
            self.conv1_2=GCNConv(128, self.hidden)
        else:
            self.conv1 = GCNConv(self.num_features, self.hidden)
        if  self.pool_method=="SAGPool":
            print("SAGPool")
            self.pool1 = SAGPool(self.hidden, ratio=self.pooling_ratio)
        elif self.pool_method=="gPool":
            print("gPool")
            self.pool1 = gPool(self.hidden, ratio=self.pooling_ratio)
        elif self.pool_method=="edgepool":
            print("edgepool")
            self.pool1 = EdgePooling(self.hidden)
        elif self.pool_method=="DiffPool":
            print("DiffPool")
            self.pool1 = DiffPool(self.hidden,ratio=self.pooling_ratio) 
            
            
        
        if args.two_conv==True:
            self.conv2 = GCNConv(self.hidden, 128)
            self.conv2_2=GCNConv(128, self.hidden)
        else:
            self.conv2 = GCNConv(self.hidden, self.hidden)
        if  self.pool_method=="SAGPool":
            self.pool2 = SAGPool(self.hidden, ratio=self.pooling_ratio)
        elif self.pool_method=="gPool":
            self.pool2 = gPool(self.hidden, ratio=self.pooling_ratio)
        elif self.pool_method=="edgepool":
            self.pool2 = EdgePooling(self.hidden)
        elif self.pool_method=="DiffPool":
            self.pool2 = DiffPool(self.hidden, ratio=self.pooling_ratio)    
        
        if args.two_conv==True:
            self.conv3 = GCNConv(self.hidden,128)
            self.conv3_2=GCNConv(128, self.hidden)
        else:
            self.conv3 = GCNConv(self.hidden,self.hidden)
            
            
        if  self.pool_method=="SAGPool":
            self.pool3 = SAGPool(self.hidden, ratio=self.pooling_ratio)
        elif self.pool_method=="gPool":
            self.pool3 = gPool(self.hidden, ratio=self.pooling_ratio)
        elif self.pool_method=="edgepool":
            self.pool3 = EdgePooling(self.hidden)
        elif self.pool_method=="DiffPool":
            self.pool3 = DiffPool(self.hidden, ratio=self.pooling_ratio)
            
            
        self.lin1 = torch.nn.Linear(self.hidden*2, self.hidden)
        self.lin2 = torch.nn.Linear(self.hidden, self.hidden//2)
        self.lin3 = torch.nn.Linear(self.hidden//2, self. num_classes)

    def forward(self, data,edge_index=None,batch=None):
        
        if not torch.is_tensor(edge_index):
            x, edge_index, batch = data.x, data.edge_index, data.batch#batch各图所对应的index
                
        else:
            x=data
            edge_index=edge_index
            batch=torch.zeros([x.shape[0]], dtype=torch.int64,device="cuda:0")
        x = F.leaky_relu(self.conv1(x, edge_index))
        if self.args.two_conv==True:
            x=F.leaky_relu(self.conv1_2(x, edge_index))
        x, edge_index, _, batch, _ = self.pool1(x, edge_index, edge_attr=None, batch=batch,layer="layer1")
        x_output1 = torch.cat([mean_pool(x, batch), max_pool(x, batch  )], dim=1)
        '''
        print("pool layer1")
        print(x.size())
        print(batch.size())
        print(x)
        print(batch)
        '''
        
        x = F.leaky_relu(self.conv2(x, edge_index))
        if self.args.two_conv==True:
            x=F.leaky_relu(self.conv2_2(x, edge_index))
        x, edge_index, _, batch, _ = self.pool2(x, edge_index, None, batch,layer="layer2")
        x_output2 = torch.cat([mean_pool(x, batch), max_pool(x, batch)], dim=1)
        '''
        print("pool layer2")
        print(x.size())
        print(batch.size())
        print(x)
        print(batch)
        '''
        x = F.leaky_relu(self.conv3(x, edge_index))
        if self.args.two_conv==True:
            x=F.leaky_relu(self.conv3_2(x, edge_index))
        x, edge_index, _, batch, _ = self.pool3(x, edge_index, None, batch,layer="layer3")

        x_output3 = torch.cat([mean_pool(x, batch), max_pool(x, batch)], dim=1)
        '''
        d=torch.cat((self.coefficient_a,self.coefficient_b,self.coefficient_c))
        temp=F.normalize(d,p=1,dim=0)
        self.coefficient_a=temp[0]
        self.coefficient_b=temp[1]
        self.coefficient_c=temp[2]
        '''
        #x = self.coefficient_a*x_output1 + self.coefficient_b*x_output2 + self.coefficient_c*x_output3
        x=x_output1 + x_output2 + x_output3
        x = F.relu(self.lin1(x))
        #x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.log_softmax(self.lin3(x), dim=-1)

        return x
