from NeuroGraph.datasets import NeuroGraphDataset
import argparse
import torch
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
import os,random
import os.path as osp
import sys
import time
from utils import *


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='HCPGender')
parser.add_argument('--seed', type=int, default=15)
parser.add_argument('--model', type=str, default='AirGNN')
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--lambda_amp', type=float, default=0.1)
parser.add_argument('--lcc', type=str2bool, default=False)
parser.add_argument('--normalize_features', type=str2bool, default=True)
parser.add_argument('--random_splits', type=str2bool, default=False)
parser.add_argument('--runs', type=int, default=10)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--early_stopping', type=int, default=100)
parser.add_argument('--hidden', type=int, default=64)
parser.add_argument('--dropout', type=float, default=0.8, help="dropout")
parser.add_argument('--K', type=int, default=10, help="the number of propagagtion in AirGNN")
parser.add_argument('--model_cache', type=str2bool, default=False)

parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--device', type=str, default='cuda')


def get_neurograph_dataset(name, normalize_features=False, transform=None):
    path = './data/airgnn'
    dataset = NeuroGraphDataset(root=path, name=name)

    if transform is not None and normalize_features:
        dataset.transform = T.Compose([T.NormalizeFeatures(), transform])
    elif normalize_features:
        dataset.transform = T.NormalizeFeatures()
    elif transform is not None:
        dataset.transform = transform

    return dataset

def prepare_data(args, lcc=False):
    transform = T.ToSparseTensor()
    dataset = get_neurograph_dataset(args.dataset, args.normalize_features, transform)
    return dataset

def main():
    args = parser.parse_args()
    print('arg : ', args)
    dataset = prepare_data(args, lcc=args.lcc)
    model = AirGNN(dataset, args)
    labels = [d.y.item() for d in dataset]

    train_tmp, test_indices = train_test_split(list(range(len(labels))), test_size=0.2, stratify=labels,random_state=123,shuffle= True)
    tmp = dataset[train_tmp]
    train_labels = [d.y.item() for d in tmp]
    train_indices, val_indices = train_test_split(list(range(len(train_labels))),
     test_size=0.125, stratify=train_labels,random_state=123,shuffle = True)
    train_dataset = tmp[train_indices]
    val_dataset = tmp[val_indices]
    test_dataset = dataset[test_indices]
    print("dataset {} loaded with train {} val {} test {} splits".format(args.dataset,len(train_dataset), len(val_dataset), len(test_dataset)))
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False)
    args.num_features,args.num_classes = dataset.num_features,dataset.num_classes
    
    criterion = torch.nn.CrossEntropyLoss()
    def train(train_loader):
        model.train()
        total_loss = 0
        for data in train_loader:  
            data = data.to(args.device)
            out = model(data)  # Perform a single forward pass.
            loss = criterion(out, data.y) 
            total_loss +=loss
            loss.backward()
            optimizer.step() 
            optimizer.zero_grad()
        return total_loss/len(train_loader.dataset)
    
    def test(loader):
        model.eval()
        correct = 0
        for data in loader:  
            data = data.to(args.device)
            out = model(data)  
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            correct += int((pred == data.y).sum())  # Check against ground-truth labels.
        return correct / len(loader.dataset), criterion(out, data.y).item()   
    
    val_acc_history, test_acc_history, test_loss_history = [],[],[]
    seeds = list(range(123, 123 + args.runs + 1))
    for index in range(args.runs):
        start = time.time()
        torch.manual_seed(seeds[index])
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seeds[index])
        random.seed(seeds[index])
        np.random.seed(seeds[index])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        gnn = eval(args.model)
        # model = GNNs(args,train_dataset,args.hidden,args.num_layers,gnn).to(args.device) # to apply GNNs 
        # model = ResidualGNNs(args,train_dataset,args.hidden,args.hidden_mlp,args.num_layers,gnn).to(args.device) ## apply GNN*
        model = AirGNN(train_dataset, args).to(args.device)
        print(model)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total number of parameters is: {total_params}")
        # model.reset_parameters()
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        loss, test_acc = [],[]
        best_val_acc,best_val_loss = 0.0,0.0
        for epoch in range(args.epochs):
            loss = train(train_loader)
            val_acc, _ = test(val_loader)
            test_acc, _ = test(test_loader)
            # if epoch%10==0:
            print("epoch: {}, loss: {}, val_acc:{}, test_acc:{}".format(epoch, np.round(loss.item(),6), np.round(val_acc,2),np.round(test_acc,2)))
            val_acc_history.append(val_acc)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), './model/airgnn/' + args.dataset+args.model+'task-checkpoint-best-acc.pkl')
            # if args.early_stopping > 0 and epoch > args.epochs // 2:
            #     tmp = tensor(val_acc_history[-(args.early_stopping + 1):-1])
            #     # print(tmp)
            #     if val_acc > tmp.mean().item():
            #         print("early stopped!")
            #         break
        model.load_state_dict(torch.load('./model/airgnn/' + args.dataset+args.model+'task-checkpoint-best-acc.pkl'))
        model.eval()
        test_acc, test_loss = test(test_loader)
        test_acc_history.append(test_acc)
        test_loss_history.append(test_loss)
    end = time.time()
    log = "dataset, {}, model, {},hidden, {},epochs, {},batch size, {}, loss,{}, acc, {}, std,{}, running time {}".format(args.dataset, args.model, 
    args.hidden, args.epochs,args.batch_size,
    round(np.mean(test_loss_history),4),round(np.mean(test_acc_history)*100,2),
    round(np.std(test_acc_history)*100,2),(end-start))
    print(log)
    log = "{}, {}, {}, {}, {},{}, {},{},{}".format(args.dataset, args.model, 
    args.hidden, args.epochs,args.batch_size,
    round(np.mean(test_loss_history),4),round(np.mean(test_acc_history)*100,2),
    round(np.std(test_acc_history)*100,2),(end-start))
    logger(log)


if __name__ == "__main__":
    main()