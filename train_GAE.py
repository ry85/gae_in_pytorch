from __future__ import division
from __future__ import print_function
from collections import defaultdict

import numpy as np
import scipy.sparse as sp
import time

import torch
import torch.nn as nn
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
              dropout=args.dropout,
              subsampling=args.subsampling)

    # if cuda move onto GPU
    if args.cuda:
        gae.cuda()
        features = features.cuda()
        data['adj_norm'] = data['adj_norm'].cuda()
        data['adj_labels'] = data['adj_labels'].cuda()
        data['features'] = data['features'].cuda()


    optimizer = optim.Adam(gae.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    #svi = SVI(gae.model, gae.guide, optimizer, loss="ELBO")
    
    # Results
    results = defaultdict(list)
    
    # Full batch training loop
    for epoch in range(args.num_epochs):

        t = time.time()
        gae.train()
        optimizer.zero_grad()

        # forward pass
        output = gae(data['features'], data['adj_norm'])

        loss.backward()
        optimizer.step()
        #print(output)

        print(gae.get_embeddings(data['features'], data['adj_norm']))

    #     # initialize loss accumulator
    #     epoch_loss = 0.
    #     # do ELBO gradient and accumulate loss
    #     epoch_loss += svi.step()
    #     # report training diagnostics
    #     if args.subsampling:
    #         normalized_loss = epoch_loss / float(2 * n_edges)
    #     else:
    #         normalized_loss = epoch_loss / (2 * N * N)

    #     # loss function
    #     #loss = gae.norm * torch.mean()
        
    #     results['train_elbo'].append(normalized_loss)

    #     # Training loss
    #     emb = gae.get_embeddings(features, adj)
    #     accuracy, roc_curr, ap_curr, = eval_gae(val_edges, val_edges_false, emb, adj_orig)
        
    #     results['accuracy_train'].append(accuracy)
    #     results['roc_train'].append(roc_curr)
    #     results['ap_train'].append(ap_curr)
        
    #     print("Epoch:", '%04d' % (epoch + 1),
    #           "train_loss=", "{:.5f}".format(normalized_loss),
    #           "train_acc=", "{:.5f}".format(accuracy), "val_roc=", "{:.5f}".format(roc_curr), "val_ap=", "{:.5f}".format(ap_curr))

    #     # Test loss
    #     if epoch % args.test_freq == 0:
    #         emb = gae.get_embeddings()
    #         accuracy, roc_score, ap_score = eval_gae(test_edges, test_edges_false, emb, adj_orig)
    #         results['accuracy_test'].append(accuracy)
    #         results['roc_test'].append(roc_curr)
    #         results['ap_test'].append(ap_curr)
    

    # print("Optimization Finished!")

    # # Test loss
    # emb = gae.get_embeddings()
    # accuracy, roc_score, ap_score = eval_gae(test_edges, test_edges_false, emb, adj_orig)
    # print('Test Accuracy: ' + str(accuracy))
    # print('Test ROC score: ' + str(roc_score))
    # print('Test AP score: ' + str(ap_score))
    
    # # Plot
    # plot_results(results, args.test_freq, path= args.dataset_str + "_results.png")

if __name__ == '__main__':

    args = dotdict()
    args.seed        = 2
    args.dropout     = 0.5
    args.num_epochs  = 200
    #args.dataset_str = 'cora'
    args.dataset_str = 'citeseer'
    args.test_freq   = 10
    args.lr          = 0.01
    args.subsampling = False
    args.weight_decay = 0.0
    args.cuda = True

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)
