from __future__ import division
from __future__ import print_function
from collections import defaultdict

import numpy as np
import scipy.sparse as sp
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from utils import load_data, dotdict, eval_gae, make_sparse, plot_results
from models import GAE
from preprocessing import mask_test_edges, preprocess_graph

def main(args):
    """ Train GAE """ 

    print("Using {} dataset".format(args.dataset_str))
    # Load data
    np.random.seed(1)
    adj, features = load_data(args.dataset_str)
    N, D = features.shape

    # Store original adjacency matrix (without diagonal entries)
    adj_orig = adj
    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)

    # Some preprocessing
    adj_train_norm   = preprocess_graph(adj_train)
    adj_train_norm   = Variable(make_sparse(adj_train_norm))
    adj_train_labels = Variable(torch.FloatTensor(adj_train + sp.eye(adj_train.shape[0]).todense()))
    features         = Variable(make_sparse(features))

    n_edges = adj_train_labels.sum()
    
    data = {
        'adj_norm'  : adj_train_norm,
        'adj_labels': adj_train_labels,
        'features'  : features,
    }

    gae = GAE(data,
              n_hidden=32,
              n_latent=16,
              dropout=args.dropout)

    # If cuda move onto GPU
    if args.cuda:
        gae.cuda()
        data['adj_norm'] = data['adj_norm'].cuda()
        data['adj_labels'] = data['adj_labels'].cuda()
        data['features'] = data['features'].cuda()

    optimizer = optim.Adam(gae.parameters(), lr=args.lr, betas=(0.95, 0.999), weight_decay=args.weight_decay)
    
    # Results
    results = defaultdict(list)
    
    # Full batch training loop
    for epoch in range(args.num_epochs):

        t = time.time()
        gae.train()
        optimizer.zero_grad()

        # forward pass
        output = gae(data['features'], data['adj_norm'])

        # Compute the loss 
        #loss = gae.norm.cuda() * torch.mean(gae.pos_weight.cuda() * criterion(output, data['adj_labels']))
        # currently no proper weighted cross entropy loss in pytorch
        # https://github.com/pytorch/pytorch/issues/5660
        logits = F.sigmoid(output)
        targets = data['adj_labels']
        max_val = (-logits).clamp(min=0)
        log_weight = 1 + (gae.pos_weight.cuda() - 1) * targets
        loss = (1 - targets) * logits + log_weight * ((-logits.abs()).exp().log1p() + max_val)
        loss = gae.norm.cuda() * torch.mean(loss)
        
        loss.backward()
        optimizer.step()

        results['train_elbo'].append(loss.data[0])

        gae.eval()
        emb = gae.get_embeddings(data['features'], data['adj_norm'])
        accuracy, roc_curr, ap_curr, = eval_gae(val_edges, val_edges_false, emb, adj_orig)
        results['accuracy_train'].append(accuracy)
        results['roc_train'].append(roc_curr)
        results['ap_train'].append(ap_curr)
        
        print("Epoch:", '%04d' % (epoch + 1),
              "train_loss=", "{:.5f}".format(loss.data[0]),
              "train_acc=", "{:.5f}".format(accuracy), "val_roc=", "{:.5f}".format(roc_curr), "val_ap=", "{:.5f}".format(ap_curr))

        # Test loss
        if epoch % args.test_freq == 0:
            gae.eval()
            emb = gae.get_embeddings(data['features'], data['adj_norm'])
            accuracy, roc_score, ap_score = eval_gae(test_edges, test_edges_false, emb, adj_orig)
            results['accuracy_test'].append(accuracy)
            results['roc_test'].append(roc_curr)
            results['ap_test'].append(ap_curr)
            gae.train()
    
    print("Optimization Finished!")

    # Test loss
    gae.eval()
    emb =  emb = gae.get_embeddings(data['features'], data['adj_norm'])
    accuracy, roc_score, ap_score = eval_gae(test_edges, test_edges_false, emb, adj_orig)
    print('Test Accuracy: ' + str(accuracy))
    print('Test ROC score: ' + str(roc_score))
    print('Test AP score: ' + str(ap_score))
    
    # Plot
    plot_results(results, args.test_freq, path= args.dataset_str + "_GAE_results.png")

if __name__ == '__main__':

    args = dotdict()
    args.seed        = 2
    args.dropout     = 0.5
    args.num_epochs  = 200
    args.dataset_str = 'cora'
    #args.dataset_str = 'citeseer'
    args.test_freq   = 10
    args.lr          = 0.01
    args.subsampling = False
    args.weight_decay = 0.0
    args.cuda = True

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)
