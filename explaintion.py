import torch, torchvision
torch.set_printoptions(edgeitems=100)
from torchvision import datasets, transforms
from torch import nn, optim
from torch.nn import functional as F
import numpy as np 
np.set_printoptions(threshold=np.inf)
import shap
import os
from torch_geometric.nn import global_mean_pool as mean_pool, global_max_pool as max_pool
from torch_geometric.data import DataLoader
import matplotlib.pyplot as plt
import networkx as nx
import copy
from torch_geometric.nn import GCNConv
from sklearn.preprocessing import normalize
def blend_color(color1, color2, f):
        r1, g1, b1 = color1
        r2, g2, b2 = color2
        r = r1 + (r2 - r1) * f
        g = g1 + (g2 - g1) * f
        b = b1 + (b2 - b1) * f
        return r, g, b
    
def draw_graph3(networkx_graph,notebook=True,output_filename='graph.html',show_buttons=True,only_physics_buttons=False):

    # import
    from pyvis import network as net
    
    # make a pyvis network
    pyvis_graph = net.Network(notebook=notebook)
    
    # for each node and its attributes in the networkx graph
    for node,node_attrs in networkx_graph.nodes(data=True):
        pyvis_graph.add_node(node,**node_attrs)
        
    # for each edge and its attributes in the networkx graph
    for source,target,edge_attrs in networkx_graph.edges(data=True):
        # if value/width not specified directly, and weight is specified, set 'value' to 'weight'
        if not 'value' in edge_attrs and not 'width' in edge_attrs and 'weight' in edge_attrs:
            # place at key 'value' the weight of the edge
            edge_attrs['value']=edge_attrs['weight']
        # add the edge
        pyvis_graph.add_edge(source,target,**edge_attrs)
        
    # turn buttons on
    if show_buttons:
        if only_physics_buttons:
            pyvis_graph.show_buttons(filter_=['physics'])
        else:
            pyvis_graph.show_buttons()
    
    # return and also save
    return pyvis_graph.show(output_filename)






class Explainer(object):#父类解释器


    def shap_values(self, X):
        raise Exception("SHAP values not implemented for this explainer!")

    def attributions(self, X):
        return self.shap_values(X)

class GradientExplainer(Explainer):#子类解释器
    
    def __init__(self, model, data, session=None, batch_size=50, local_smoothing=0):

        self.explainer = _PyTorchGradientExplainer(model, data, batch_size, local_smoothing)

    def shap_values(self, X, nsamples=200, ranked_outputs=None, output_rank_order="max", rseed=None):
        
        return self.explainer.shap_values(X, nsamples, ranked_outputs, output_rank_order, rseed)


class _PyTorchGradientExplainer(Explainer):#子类解释器

    def __init__(self, model, data,batch_size=50, local_smoothing=0):

        # 确定pytorch
        global torch
        if torch is None:
            import torch
            #if LooseVersion(torch.__version__) < LooseVersion("0.4"):
                #warnings.warn("Your PyTorch version is older than 0.4 and not supported.")

        # 检查是否为多项输入
        self.multi_input = False
        if type(data) == list:
            self.multi_input = True
        if type(data) != list:
            data = [data]
        # for consistency, the method signature calls for data as the model input.
        # However, within this class, self.model_inputs is the input (i.e. the data passed by the user)
        # and self.data is the background data for the layer we want to assign importances to. If this layer is
        # the input, then self.data = self.model_inputs
        self.model_inputs = data
        self.batch_size = batch_size
        self.local_smoothing = local_smoothing
        self.layer = None
        if type(model) == tuple:
            model, layer = model
            model = model.eval()
            self.layer = layer
            with torch.no_grad():
                _ = model(*data)
        else:
            self.data = data
        self.model = model.eval()
        multi_output = False
        outputs = self.model(*self.model_inputs)#####得到output
        if len(outputs.shape) > 1 and outputs.shape[1] > 1:
            multi_output = True
        self.multi_output = multi_output

        if not self.multi_output:
            self.gradients = [None]
        else:
            self.gradients = [None for i in range(outputs.shape[1])]

    def gradient(self, idx, inputs):
        self.model.zero_grad()
        #X = [x.requires_grad_() for x in inputs]
        outputs=[]
        grads=[]
        for x in inputs:
            #print(x.x)
            x.x.requires_grad_()
            outputs.append(self.model(x))
        
        idx=int(idx)
        for i,x in enumerate(inputs):
            selected = outputs[i][0][idx]
            #grads = [torch.autograd.grad(selected, x,retain_graph=True if idx + 1 < len(X) else None)[0].cpu().numpy()
                         #for idx, x in enumerate(X)]
            grads.append(torch.autograd.grad(selected, x.x,retain_graph=True )[0].cpu().numpy())
        #(50,351,89)
        return grads

    def shap_values(self, X, nsamples=200, ranked_outputs=None, output_rank_order="max", rseed=None):

        # X ~ self.model_input
        # X_data ~ self.data
        data1=copy.deepcopy(X)
        # check if we have multiple inputs
        if not self.multi_input:
            assert type(X) != list, "Expected a single tensor model input!"
            X = [X]
        else:
            assert type(X) == list, "Expected a list of model inputs!"
        #[Batch(batch=[351], edge_index=[2, 1584], x=[351, 89], y=[1])]
        model_output_ranks = (torch.ones((1, len(self.gradients))).int() *
                                  torch.arange(0, len(self.gradients)).int())
        #print(model_output_ranks)[[0, 1]]
        X_batches = 1
        output_phis = []
        output_phi_vars = []
        
        #torch.Size([1, 1, 28, 28])
        
        
        #samples_input = [torch.zeros((nsamples,) + X[l].shape[1:], device=X[l].device)]
        samples_input=[]
        samples_delta=[]
        for i in range(200):
            samples_input.append(copy.deepcopy(X[0]))
            samples_delta.append(copy.deepcopy(X[0].x))
        #print(samples_input)[[batch]]
        #print(len(samples_input[0]))200
        #samples_delta = [np.zeros((nsamples, ) + self.data[l].shape[1:]) for l in range(len(self.data))]
        samples_input=[samples_input]
        samples_delta=[samples_delta]

        #(1,200, 1, 28, 28)
        # use random seed if no argument givenle
        if rseed is None:
            rseed = np.random.randint(0, 1e6)

        for i in range(model_output_ranks.shape[1]):#类别数
            np.random.seed(rseed)  # so we get the same noise patterns for each output class
            phis = []
            #phi_vars = []
        
           
            #torch.Size([100, 1, 28, 28])
            phis.append(np.zeros((X_batches,) + X[0].x.shape))
            #print("phis_details")
            #phis(1, 1, 351, 89)
            number_node=int(X[0].batch.shape[0])
            for j in range(1):
                # 填充样本点
                for k in range(nsamples):
                    rind = np.random.choice(20)#150个背景中选择
                    t = np.random.uniform()#取一个位于0-1之间的均匀分布
                    x=torch.tensor(X[0].x,device='cuda:0').clone().detach()
                    #x = X[0].x
                    #计算IG对角线积分

                    #samples_input[0][k].x = 
                    if (rind+1)*number_node<self.model_inputs[0].x.shape[0]:
                        temp=copy.deepcopy((t * x + (1 - t) * (self.model_inputs[0].x[rind*number_node:(rind+1)*number_node,:])).clone().detach())
                    else:
                        rind=int(rind*0.3)
                        temp=copy.deepcopy((t * x + (1 - t) * (self.model_inputs[0].x[rind*number_node:(rind+1)*number_node,:])).clone().detach())
                        
                        
                    samples_input[0][k].x=copy.deepcopy(torch.tensor(temp,device='cuda:0'))
                    #print(samples_input[0][k].x)
                    samples_delta[0][k] = (x - (self.model_inputs[0].x[rind*number_node:(rind+1)*number_node,:]).clone().detach()).cpu().numpy()
                    #print(samples_input[0][k].x)
                    #samples_input和samples_delta经过插值扩展的(1,200, 1, 28, 28)矩阵
                # compute the gradients at all the sample points
                find = model_output_ranks[j, i]
                #model_output_ranks
                #tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
                #print(samples_input)
                grads = []
                
                    
                for b in range(0, nsamples, self.batch_size):#nsamples=200,batch_size=50
                    
                    batch = samples_input[0][b:min(b+self.batch_size,nsamples)] 
                    #for batch in batch:
                        #print(batch.x)
                        #print(batch.x.size())
                        
                    grads.append(self.gradient(find, batch))
                    #梯度计算
                #(4,1,50,1,28,28)
                #(4,50,351,89)

                #print(len(grads))
                #print(grads[0][0].shape)
                #for g in grads:
                    
                grad = [np.concatenate([g for g in grads], 0)]
                #(1,200, 1, 28, 28)
                #(1,200,351,89)
                # assign the attributions to the right part of the output arrays
                samples = grad[0] * samples_delta[0]
                phis[0][j] = samples.mean(0)
                    #(1,1,1,28,28)
                    
            output_phis.append(phis[0])
        index = np.argmax(self.model(data1).cpu().data.numpy())
        mask=np.sum( output_phis[index], axis=2)
        mask=mask[0]
        mask=mask.argsort()[::-1] 
            
        return mask


    
    
    
class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers,add_layer=False):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []
        self.add_layer=add_layer

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, data):
        outputs = []
        self.gradients = []
        x, edge_index, batch = data.x, data.edge_index, data.batch
        #for name, module in self.model._modules.items():#生成每一层结果，迭代生成所需pooling层的结果
        x=F.leaky_relu(self.model._modules["conv1"](x, edge_index))
        if self.add_layer==True:
            x=F.leaky_relu(self.model._modules["conv1_2"](x, edge_index))
        outputs1=x
        
        x, edge_index, _, batch, _ =self.model._modules["pool1"](x, edge_index, None, batch,layer="layer1")
        x.register_hook(self.save_gradient)
        x_1=x
        x_output1 = torch.cat([mean_pool(x, batch), max_pool(x, batch  )], dim=1)
        x=F.leaky_relu(self.model._modules["conv2"](x, edge_index))
        if self.add_layer==True:
            x=F.leaky_relu(self.model._modules["conv2_2"](x, edge_index))
        outputs2=x
        x, edge_index, _, batch, _ =self.model._modules["pool2"](x, edge_index, None, batch,layer="layer2")
        x.register_hook(self.save_gradient)        

        x_2=x
        x_output2 = torch.cat([mean_pool(x, batch), max_pool(x, batch  )], dim=1)
        x=F.leaky_relu(self.model._modules["conv3"](x, edge_index))
        if self.add_layer==True:
            x=F.leaky_relu(self.model._modules["conv3_2"](x, edge_index))
        
        outputs3=x
        x, edge_index, _, batch, _ =self.model._modules["pool3"](x, edge_index, None, batch,layer="layer3")
        x.register_hook(self.save_gradient)
        x_3=x
        x_output3 = torch.cat([mean_pool(x, batch), max_pool(x, batch  )], dim=1)
        x = x_output1 + x_output2 + x_output3
        
        return outputs1,outputs2,outputs3,x_1,x_2,x_3,x#输出卷积层的结果
           
            #x = module()
            #if name in self.target_layers:
                #x.register_hook(self.save_gradient)
                #outputs += [x]
            
class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, target_layers,add_layer=False):
        self.model = model
        self.feature_extractor = FeatureExtractor(self.model, target_layers,add_layer=False)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        outputs1,outputs2,outputs3,x_1,x_2,x_3, output = self.feature_extractor(x)

        output = output.view(output.size(0), -1)
        #output = self.model.classifier(output)#获得输出结果
        output = F.relu(self.model._modules["lin1"](output))
        #x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        output = F.relu(self.model._modules["lin2"](output))
        output = F.log_softmax(self.model._modules["lin3"](output), dim=-1)
         
        return outputs1,outputs2,outputs3,x_1,x_2,x_3, output

    
    
    
class GradCam:
    def __init__(self, model, target_layer_names,add_layer=False):
        self.model = model
        self.model.eval()
        self.model = model.cuda()
        self.extractor = ModelOutputs(self.model, target_layer_names,add_layer=False)
        self.target_layer=target_layer_names
        self.conv_layer = GCNConv(1,1)
    def forward(self, input):
        return self.model(input)
    
    def __call__(self, input, index=None):
        
        def mapping_two_index(index1,index2):#mapping index1 into index2
            for i in range(len(index2)):
                index2[i]=index1[index2[i]]
            return index2
        def mapping_index(mask,input_x,index):#return the matrix that cotained the importance which is the same size as the original input
            input_x=input_x.cpu().data.numpy()
            input_x1=np.zeros((input_x.shape[0],input_x.shape[1]))
            for i in range(mask.shape[0]):
                input_x1[index[i],np.where(input_x[index[i],:]==1)]=mask[i]
            return input_x1
        def cal_distance_matrix(feature_matrix):#calculate each node's features relationship bwtween each other
            matrix_1=feature_matrix[:,1:]
            matrix_2=feature_matrix[:,:feature_matrix.shape[1]-1]
            distance_matrix=(matrix_1/matrix_2)
            return distance_matrix
        def copying_value(x, edge_index,final_mask,index):
            mask=np.zeros(x.shape[0])
            for j,i in enumerate(index):
                mask[i]=final_mask[j]
                temp_truple=np.where(edge_index==i)
                for t in range(int(temp_truple[1].shape[0]/2)):
                    mask[edge_index[1][temp_truple[1][t]]]=final_mask[j]
            return mask
        def cal_similarity(feature_matrix,output_matrix):
            index=[]
            for i in range(output_matrix.shape[0]):
                temp_1=output_matrix[i,:]
                distance=0
                for j in range(feature_matrix.shape[0]):
                    temp_2=feature_matrix[j,:]
                    distance_temp=np.dot(temp_1,temp_2)/(np.linalg.norm(temp_1) * np.linalg.norm(temp_2))
                    if np.abs(distance_temp)>=distance:
                        distance=np.abs(distance_temp)
                        index_temp=j
                    else:
                        pass
                index.append(index_temp)
            return index
        
        
        features1,features2,features3,x_1,x_2,x_3,output = self.extractor(input)#计算卷积层的特征以及mlp的线性输出
        if index == None:
            index = np.argmax(output.cpu().data.numpy())
        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)
        #self.model.zero_grad()#梯度置零
        #self.model.classifier.zero_grad()
        one_hot.backward(retain_graph=True)
        features1=features1.cpu().data.numpy()
        features2=features2.cpu().data.numpy()
        features3=features3.cpu().data.numpy()
        x_1=x_1.cpu().data.numpy()
        x_2=x_2.cpu().data.numpy()
        x_3=x_3.cpu().data.numpy()
        
        
        if self.target_layer==1:
            x, edge_index = input.x.cpu().data.numpy(), input.edge_index.cpu().data.numpy()
            grads_val = self.extractor.get_gradients()[2].cpu().data.numpy()
       # grads_val = self.extractor.get_gradients()
            grads_val=np.sum(grads_val,axis=0)#add each column
            cam=np.zeros((x_1.shape[0],x_1.shape[1]))
            #grads_val=np.sum(grads_val,axis=1)
            for i,grad in enumerate(grads_val):
                cam[:,i]=grad*x_1[:,i]
            #cam=grads_val*x_1
            
            mask=np.sum(cam, axis=1)#add each row
            #x_1=cal_distance_matrix(x_1)
            #features1=cal_distance_matrix(features1)
            index=cal_similarity(features1,x_1)
            final_mask=copying_value(x, edge_index,mask,index)
            
            final_mask=final_mask.argsort()[::-1] 
            return final_mask
            
            
            #final_mask=mapping_index(mask,input.x,index)
            #final_mask = torch.from_numpy(final_mask).float().to("cuda:0")
            #final_mask=F.leaky_relu(self.model._modules["conv1"](final_mask, input.edge_index))
            #final_mask=final_mask.cpu().data.numpy()
            #final_mask=np.sum(final_mask, axis=1)
            #final_mask=final_mask.argsort()[::-1] 
            #return final_mask
        
        elif self.target_layer==2:
            x, edge_index = input.x.cpu().data.numpy(), input.edge_index.cpu().data.numpy()
            grads_val2 = self.extractor.get_gradients()[1].cpu().data.numpy()
            cam=np.zeros((x_2.shape[0],x_2.shape[1]))
            for i,grad in enumerate(grads_val2):
                cam[i]=grad*x_2[i]
            mask=np.sum(cam, axis=1)
            #x_1=cal_distance_matrix(x_1)
            #features1=cal_distance_matrix(features1)
            index1=cal_similarity(features1,x_1)
            
            #x_2=cal_distance_matrix(x_2)
            #features2=cal_distance_matrix(features2)
            index2=cal_similarity(features2,x_2)
            index2=mapping_two_index(index1,index2)
            final_mask=copying_value(x, edge_index,mask,index2)
            #final_mask = torch.from_numpy(final_mask).float().to("cuda:0")
            #final_mask=F.leaky_relu(self.model._modules["conv1"](final_mask, input.edge_index))
            #final_mask=final_mask.cpu().data.numpy()
            #final_mask=np.sum(final_mask, axis=1)
            final_mask=final_mask.argsort()[::-1] 
        
            return final_mask
        elif self.target_layer==3:
            x, edge_index = input.x.cpu().data.numpy(), input.edge_index.cpu().data.numpy()
            grads_val3 = self.extractor.get_gradients()[0].cpu().data.numpy()
            cam=np.zeros((x_3.shape[0],x_3.shape[1]))
            for i,grad in enumerate(grads_val3):
                cam[i]=grad*x_3[i]
            mask=np.sum(cam, axis=1)
            #x_1=cal_distance_matrix(x_1)
            #features1=cal_distance_matrix(features1)
            index1=cal_similarity(features1,x_1)
            
            #x_2=cal_distance_matrix(x_2)
            #features2=cal_distance_matrix(features2)
            index2=cal_similarity(features2,x_2)
            
            index2=mapping_two_index(index1,index2)
            
            #x_3=cal_distance_matrix(x_3)
            #features3=cal_distance_matrix(features3)
            index3=cal_similarity(features3,x_3)
            
            
            index3=mapping_two_index(index2,index3)
            
            
            final_mask=copying_value(x, edge_index,mask,index3)
            #final_mask = torch.from_numpy(final_mask).float().to("cuda:0")
            #final_mask=F.leaky_relu(self.model._modules["conv1"](final_mask, input.edge_index))
            #final_mask=final_mask.cpu().data.numpy()
            #final_mask=np.sum(final_mask, axis=1)
            final_mask=final_mask.argsort()[::-1] 
            return final_mask
        #cam=grads_val.mul(features)
        #print(cam)
        #grads_val = np.array(self.extractor.get_gradients())
        
        '''
        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)


        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam
        '''