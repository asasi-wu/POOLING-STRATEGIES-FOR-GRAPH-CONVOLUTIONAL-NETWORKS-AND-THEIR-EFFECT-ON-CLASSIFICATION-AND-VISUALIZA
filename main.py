import torch
from torch_geometric.datasets import TUDataset
import config
import os
import numpy as np
from scipy.interpolate import make_interp_spline
from torch.utils.data import random_split
from torch.utils.data import ConcatDataset
from torch_geometric.data import DataLoader
from model_selection import model_Selection
import train
import matplotlib.pyplot as plt
import copy
import tester as test
from torch_geometric.utils import degree
class MyFilter(object):
    def __call__(self, data):
        return 100<=data.num_nodes <= 1000
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
#import the module
def plot_result(acc_list,loss_list,typedef):
    x = np.arange(len(acc_list))

    plt.subplot(2, 1, 1)
    xnew = np.linspace(x.min(),x.max(),500)
 
    acc_list_smooth = make_interp_spline(x, acc_list)(xnew)
    
    plt.plot(xnew, acc_list_smooth, '-')
    plt.title('{} accuracy vs. epoches'.format(typedef))
    plt.ylabel('{} accuracy'.format(typedef))
    plt.subplot(2, 1, 2)
    
    loss_list_smooth = make_interp_spline(x, loss_list)(xnew)
    plt.plot(xnew, loss_list_smooth, '-')
    plt.xlabel('epoches')
    plt.ylabel('{} loss'.format(typedef))
    plt.show()
    plt.savefig("accuracy_loss.jpg")


class add_degree(object):
    r"""Adds a constant value to each node feature.

    Args:
        value (int, optional): The value to add. (default: :obj:`1`)
        cat (bool, optional): If set to :obj:`False`, all existing node
            features will be replaced. (default: :obj:`True`)
    """

    def __init__(self, value=1, cat=True):
        self.value = value
        self.cat = cat

    def __call__(self, data):
        x = data.x
        
        c = torch.full((data.num_nodes, 1), self.value)

        if x is not None and self.cat:
            x = x.view(-1, 1) if x.dim() == 1 else x
            data.x = torch.cat([x, c.to(x.dtype).to(x.device)], dim=-1)
            data.x[:,-1]=torch.unsqueeze(degree(data.edge_index[0]),0)
        else:
            data.x = c

        return data

    def __repr__(self):
        return '{}(value={})'.format(self.__class__.__name__, self.value)

    


                    
#def hyperparameters_search(args):

'''
number_training = int(len(dataset)*0.2)*4
number_val = int(len(dataset)*0.2)
number_test = len(dataset) - number_training
number_list=[]
dataset_list=[]
for i in range(5):
    if i == 4:
        number_list.append(number_test)
    else:
        number_list.append(number_val)
print(number_list)
dataset_list=list(random_split(dataset,number_list))

for i in range(5):
    best_parameters=[]
    
    #training_set,validation_set,test_set = random_split(dataset,[number_training,number_val,number_test])
    print(i)
    test_set=dataset_list[i]
 
    temp=dataset_list.copy()
    temp.pop(i)
    training_set=ConcatDataset(temp)
    number_traintemp = int(len(training_set)*0.8)
    number_testtemp = len(training_set)-number_traintemp
    training_set,validation_set=random_split(training_set,[number_traintemp,number_testtemp])

    train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(validation_set,batch_size=args.batch_size,shuffle=False)
    test_loader = DataLoader(test_set,batch_size=1,shuffle=False)
    
    #best_parameters.append(model_Selection(args,train_loader,val_loader,test_loader))
    acctest_list,losstest_list,accval_list,lossval_list=train.train_process(args,train_loader,val_loader,test_loader)
    #plot_result( acctest_list,losstest_list,"test")
    plot_result(accval_list,lossval_list,"validation")
    
'''
"""
acc_list=[]
loss_list=[]

for i in ["SAGPool","gPool","edgepool","DiffPool"]:
    for j in range(4):
        print(j)
        args = config.get_config()
        torch.manual_seed(args.seed)
        dataset = TUDataset(os.path.join('data',args.dataset),name=args.dataset,pre_filter=MyFilter(),use_node_attr=True)
        args.num_classes = dataset.num_classes
        args.num_features = dataset.num_features
        num_training = int(len(dataset)*0.8)
        num_val = int(len(dataset)*0.1)
        num_test = len(dataset) - (num_training+num_val)
        training_set,validation_set,test_set = random_split(dataset,[num_training,num_val,num_test])
        train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(validation_set,batch_size=args.batch_size,shuffle=False)
        test_loader = DataLoader(test_set,batch_size=1,shuffle=False)
        args.pool_method=i
        accval_list,lossval_list=train.train_process(args,train_loader,val_loader,test_loader)
        acc_list.append(accval_list)
        loss_list.append(lossval_list)
print(acc_list)
print(loss_list)
#plot_result(acc_list,loss_list,"validation")    

"""
args = config.get_config()
torch.manual_seed(args.seed)
for data in ["DD","PROTEINS","ENZYMES"]:
    args.dataset=data
    for pooling_methods in ["SAGPool","gPool","edgepool","DiffPool"]:
        args.pool_method=pooling_methods
        dataset = TUDataset(os.path.join('data',args.dataset),name=args.dataset,pre_filter=MyFilter(),use_node_attr=True)
        number_training = int(len(dataset)*0.25)*3
        number_val = int(len(dataset)*0.25)
        number_test = len(dataset) - number_training
        number_list=[]
        dataset_list=[]
        for i in range(4):
            if i == 3:
                number_list.append(number_test)
            else:
                number_list.append(number_val)
        
        cross_loss=[]
        cross_accuracy=[]
        cross_test_accuracy=[]
        for i in range(4):
            
            best_accuracy=0
            best_add_node=None
            best_batch_size=0
            best_hidden_size=0
            best_learning_rate=0
            best_con_number=None
            best_patience=0
            for add_node in [True,False]:
                args.add_node=add_node
                if not args.add_node:
                    dataset = TUDataset(os.path.join('data',args.dataset),name=args.dataset,pre_filter=MyFilter(),use_node_attr=True)
                else:
                    dataset = TUDataset(os.path.join('data',args.dataset),name=args.dataset,transform=add_degree(value=1),pre_filter=MyFilter(),use_node_attr=True)
                args.num_classes = dataset.num_classes
                args.num_features = dataset.num_features
                dataset_list=list(random_split(dataset,number_list)) 
                test_set=dataset_list[i]
                temp=dataset_list.copy()
                temp.pop(i)
                training_set=ConcatDataset(temp)
                number_trainselect = int(len(training_set)*0.6)
                number_testselect = len(training_set)-number_trainselect
                training_select,validation_select=random_split(training_set,[number_trainselect,number_testselect])
                #the above is preparing the datasetfor 
                for learning_rate in [0.01,0.001]:
                #for learning_rate in [0.01]:
                    args.lr=learning_rate
                    #for batch_size in [64]:
                    for batch_size in [64,32]: 
                        print(batch_size)
                        args.batch_size=batch_size
                        for hidden_size in [64,128]:
                        #for hidden_size in [128]:
                            args.hidden_size=hidden_size
                            for patience in [15,30]:
                            #for patience in [20]:
                                args.patience=patience
                                for twoconv in [False,True]:
                                #for twoconv in [False]:
                                    args.two_conv=twoconv
                                    train_loader = DataLoader(training_select, batch_size=args.batch_size, shuffle=True)
                                    val_loader = DataLoader(validation_select,batch_size=args.batch_size,shuffle=False)
                                    accval_list,lossval_list=train.train_process(args,train_loader,val_loader)
                                    if max(accval_list)>best_accuracy:
                                        best_accuracy=max(accval_list)
                                        best_add_node=args.add_node
                                        best_batch_size=args.batch_size
                                        best_hidden_size=args.hidden_size
                                        best_learning_rate=args.lr
                                        best_con_number=args.two_conv
                                        best_patience=args.patience
                                        
                                        
                                        
            args.add_node=best_add_node
            args.batch_size=best_batch_size
            args.hidden_size=best_hidden_size
            args.lr=best_learning_rate
            args.two_conv=best_con_number
            args.patience=best_patience
            '''
            print('add_degree:{}'.format(args.add_node)+'\n')
            print('batch_size:{}'.format(args.batch_size)+'\n')
            print('hidden_size:{}'.format(args.hidden_size)+'\n')
            print('learning_rate:{}'.format( args.lr)+'\n')
            print('two_convolution_layer:{}'.format(args.two_conv)+'\n')
            print('patience:{}'.format(args.patience)+'\n')
            '''
            if not args.add_node:
                dataset = TUDataset(os.path.join('data',args.dataset),name=args.dataset,pre_filter=MyFilter(),use_node_attr=True)
            else:
                dataset = TUDataset(os.path.join('data',args.dataset),name=args.dataset,transform=add_degree(value=1),pre_filter=MyFilter(),use_node_attr=True)  
            args.num_classes = dataset.num_classes
            args.num_features = dataset.num_features
            dataset_list=list(random_split(dataset,number_list)) 
            test_set=dataset_list[i]
            temp=dataset_list.copy()
            temp.pop(i)
            training_set=ConcatDataset(temp)
            test_loader = DataLoader(test_set,batch_size=1,shuffle=False)
            number_trainevalue = int(len(training_set)*0.8)
            number_testevalue = len(training_set)-number_trainevalue
            training_evalue,validation_evalue=random_split(training_set,[number_trainevalue,number_testevalue])
            train_loader = DataLoader(training_evalue, batch_size=args.batch_size, shuffle=True)   
            val_loader = DataLoader(validation_evalue,batch_size=args.batch_size,shuffle=False)        
            accval_list,lossval_list=train.train_process(args,train_loader,val_loader,cross_number=i)
            cross_loss.append(lossval_list)
            cross_accuracy.append(accval_list)
            model=torch.load('{}_model_{}_cross_vilidation{}.pkl'.format(args.dataset,args.pool_method,i))
            cross_test_accuracy.append(test.testing(model,test_loader,args.device))
            f=open('{}_model_{}_cross{}_best.txt'.format(args.dataset,args.pool_method,i),'w')
            f.write('add_degree:{}'.format(args.add_node)+'\n')
            f.write('batch_size:{}'.format(args.batch_size)+'\n')
            f.write('hidden_size:{}'.format(args.hidden_size)+'\n')
            f.write('learning_rate:{}'.format( args.lr)+'\n')
            f.write('two_convolution_layer:{}'.format(args.two_conv)+'\n')
            f.write('patience:{}'.format(args.patience)+'\n')
            f.close()
            
        f=open('{}_model_{}_accuracy.txt'.format(args.dataset,args.pool_method),'w')
        for accuracy in cross_test_accuracy:
            f.write(str(accuracy)+'\n')
        f.close()
        
        f=open('{}_model_{}_cross_accuracy.txt'.format(args.dataset,args.pool_method),'w')
        for accuracy in cross_accuracy:
            f.write(str(accuracy)+'\n')
        f.close()
        f=open('{}_model_{}_cross_loss.txt'.format(args.dataset,args.pool_method),'w')
        for loss in cross_loss:
            f.write(str(loss)+'\n')
        f.close()