#set the args
import numpy as np
import train
def model_Selection(args,train_loader,val_loader,test_loader):
    best_hidden_size=0
    best_lr=0.
    best_weight=0.
    best_acc=0.
    for hidden in range(20,50,10):
     
        args.hidden_size=hidden
        for lr in np.arange(0.001,0.002,0.0005):
            args.lr=lr
            for weight_decay in np.arange(0.0001,0.0002,0.00005):
              
                args.weight_decay=weight_decay
                temp=train.train_process(args,train_loader,val_loader,test_loader)
                if temp>=best_acc:                   
                        best_hidden_size=args.hidden_size
                        best_lr=args.lr
                        best_weight=args.weight_decay
                        best_acc=temp
                   
    return  best_hidden_size,best_lr, best_weight,best_acc