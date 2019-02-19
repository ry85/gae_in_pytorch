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

from utils import load_data, dotdict, eval_gae, make_sparse, plot_results
from models import GVAE
from preprocessing import mask_test_edges, preprocess_graph

def main(args):
    """ Train GAE """ 

    # Compute the device upon which to run
    device = torch.device("cuda" if args.use_cuda else "cpu")

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
    adj_train_norm   = make_sparse(adj_train_norm)
    adj_train_labels = torch.FloatTensor(adj_train + sp.eye(adj_train.shape[0]).todense())
    features         = make_sparse(features)

    n_edges = adj_train_labels.sum()
    
    data = {
        'adj_norm'  : adj_train_norm,
        'adj_labels': adj_train_labels,
        'features'  : features,
    }

    gae = GVAE(data,
              n_hidden=32,
              n_latent=16,
              dropout=args.dropout)

    # Send the model and data to the available device
    gae.to(device)
    data['adj_norm'] = data['adj_norm'].to(device)
    data['adj_labels'] = data['adj_labels'].to(device)
    data['features'] = data['features'].to(device)

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

        # Compute the loss ------------------------------------------

        # Compute the weighted_cross_entropy_with_logits ------------
        logits = output
        targets = data['adj_labels']
        cost = gae.norm * F.binary_cross_entropy_with_logits(logits, targets, pos_weight=gae.pos_weight)

        # compute the latent loss -----------------------------------
        log_lik = cost
        #self.kl = (0.5 / N) * tf.reduce_mean(tf.reduce_sum(1 + 2 * model.z_log_std - tf.square(model.z_mean) - tf.square(tf.exp(model.z_log_std)), 1))
        kl = (0.5 / N) * torch.mean(torch.sum(1 + 2 * gae.log_sig - torch.mul(gae.mu, gae.mu) - torch.mul(torch.exp(gae.log_sig), torch.exp(gae.log_sig)), 1))
        cost -= kl
        loss = cost
        
        loss.backward()
        optimizer.step()

        results['train_elbo'].append(loss.item())

        gae.eval()
        emb = gae.get_embeddings(data['features'], data['adj_norm'])
        accuracy, roc_curr, ap_curr, = eval_gae(val_edges, val_edges_false, emb, adj_orig)
        results['accuracy_train'].append(accuracy)
        results['roc_train'].append(roc_curr)
        results['ap_train'].append(ap_curr)
        
        print("Epoch:", '%04d' % (epoch + 1),
              "train_loss=", "{:.5f}".format(loss.item()),
              "train_acc=", "{:.5f}".format(accuracy), "val_roc=", "{:.5f}".format(roc_curr), "val_ap=", "{:.5f}".format(ap_curr))

        # Test loss
        if epoch % args.test_freq == 0:
            with torch.no_grad():
                gae.eval()
                emb = gae.get_embeddings(data['features'], data['adj_norm'])
                accuracy, roc_score, ap_score = eval_gae(test_edges, test_edges_false, emb, adj_orig)
                results['accuracy_test'].append(accuracy)
                results['roc_test'].append(roc_curr)
                results['ap_test'].append(ap_curr)
            gae.train()
    
    print("Optimization Finished!")
    
    with torch.no_grad():
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
    args.use_cuda = False

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)
