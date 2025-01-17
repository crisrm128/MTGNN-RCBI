import argparse
import math
import time

import torch
import torch.nn as nn
from net import gtnet
import numpy as np
import importlib

from util import *
from trainer import Optim

np.random.seed(42) # To allow reproducibility

def evaluate(data, X, Y, model, evaluateL2, evaluateL1, batch_size):
    model.eval()
    total_loss = 0
    total_loss_l1 = 0
    n_samples = 0
    predict = None
    test = None

    for X, Y in data.get_batches(X, Y, batch_size, False):
        X = torch.unsqueeze(X,dim=1)
        X = X.transpose(2,3)
        with torch.no_grad():
            output = model(X)
        output = torch.squeeze(output)
        if len(output.shape)==1:
            output = output.unsqueeze(dim=0)
        if predict is None:
            predict = output
            test = Y
        else:
            predict = torch.cat((predict, output))
            test = torch.cat((test, Y))

        scale = data.scale.expand(output.size(0), data.m)
        total_loss += evaluateL2(output * scale, Y * scale).item()
        total_loss_l1 += evaluateL1(output * scale, Y * scale).item()
        n_samples += (output.size(0) * data.m)

    rse = math.sqrt(total_loss / n_samples) / data.rse
    rae = (total_loss_l1 / n_samples) / data.rae

    predict = predict.data.cpu().numpy()
    Ytest = test.data.cpu().numpy()
    sigma_p = (predict).std(axis=0)
    sigma_g = (Ytest).std(axis=0)
    mean_p = predict.mean(axis=0)
    mean_g = Ytest.mean(axis=0)
    index = (sigma_g != 0)
    correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
    correlation = (correlation[index]).mean()
    return rse, rae, correlation, predict # Modified to return the predicted values


def train(data, X, Y, model, criterion, optim, batch_size):
    model.train()
    total_loss = 0
    n_samples = 0
    iter = 0
    for X, Y in data.get_batches(X, Y, batch_size, True):
        model.zero_grad()
        X = torch.unsqueeze(X,dim=1)
        X = X.transpose(2,3)
        if iter % args.step_size == 0:
            perm = np.random.permutation(range(args.num_nodes))
        num_sub = int(args.num_nodes / args.num_split)

        for j in range(args.num_split):
            if j != args.num_split - 1:
                id = perm[j * num_sub:(j + 1) * num_sub]
            else:
                id = perm[j * num_sub:]
            id = torch.tensor(id).to(device)
            tx = X[:, :, id, :]
            ty = Y[:, id]
            output = model(tx,id)
            output = torch.squeeze(output)
            scale = data.scale.expand(output.size(0), data.m)
            scale = scale[:,id]
            loss = criterion(output * scale, ty * scale)
            loss.backward()
            total_loss += loss.item()
            n_samples += (output.size(0) * data.m)
            grad_norm = optim.step()

        if iter%100==0:
            print('iter:{:3d} | loss: {:.3f}'.format(iter,loss.item()/(output.size(0) * data.m)))
        iter += 1
    return total_loss / n_samples

# Input arguments ----------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
parser.add_argument('--data', type=str, default='./data/solar_AL.txt',
                    help='location of the data file')
parser.add_argument('--log_interval', type=int, default=2000, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model/model.pt',
                    help='path to save the final model')
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--L1Loss', type=bool, default=True)
parser.add_argument('--normalize', type=int, default=2)
parser.add_argument('--device',type=str,default='cuda:1',help='')
parser.add_argument('--gcn_true', type=bool, default=True, help='whether to add graph convolution layer')
parser.add_argument('--buildA_true', type=bool, default=True, help='whether to construct adaptive adjacency matrix')
parser.add_argument('--gcn_depth',type=int,default=2,help='graph convolution depth')
parser.add_argument('--num_nodes',type=int,default=137,help='number of nodes/variables')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--subgraph_size',type=int,default=20,help='k')
parser.add_argument('--node_dim',type=int,default=40,help='dim of nodes')
parser.add_argument('--dilation_exponential',type=int,default=2,help='dilation exponential')
parser.add_argument('--conv_channels',type=int,default=16,help='convolution channels')
parser.add_argument('--residual_channels',type=int,default=16,help='residual channels')
parser.add_argument('--skip_channels',type=int,default=32,help='skip channels')
parser.add_argument('--end_channels',type=int,default=64,help='end channels')
parser.add_argument('--in_dim',type=int,default=1,help='inputs dimension')
parser.add_argument('--seq_in_len',type=int,default=24*7,help='input sequence length')
parser.add_argument('--seq_out_len',type=int,default=1,help='output sequence length')
parser.add_argument('--horizon', type=int, default=3)
parser.add_argument('--layers',type=int,default=5,help='number of layers')

parser.add_argument('--batch_size',type=int,default=32,help='batch size')
parser.add_argument('--lr',type=float,default=0.0001,help='learning rate')
parser.add_argument('--weight_decay',type=float,default=0.00001,help='weight decay rate')

parser.add_argument('--clip',type=int,default=5,help='clip')

parser.add_argument('--propalpha',type=float,default=0.05,help='prop alpha')
parser.add_argument('--tanhalpha',type=float,default=3,help='tanh alpha')

parser.add_argument('--epochs',type=int,default=1,help='')
parser.add_argument('--num_split',type=int,default=1,help='number of splits for graphs')
parser.add_argument('--step_size',type=int,default=100,help='step_size')

# New ones added for setting the number of experiments and allowing the sensitivity analysis
parser.add_argument('--exp_num',type=int, default=3, help='number of runs/experiments')
parser.add_argument('--sensitivity', type=bool, default=False, help='usage of sensitivity analysis')
parser.add_argument('--sensitivity_num', type=int, default=1, help='number of variables used for sensitivity analysis')

args = parser.parse_args()
device = torch.device(args.device)
torch.set_num_threads(3)

# ----------------------------------------------------------------------------------------------------------------------

# Function that performs the original training within the main function. Now adjusted for the sensitivity analysis.
def main_training(Data):
    model = gtnet(args.gcn_true, args.buildA_true, args.gcn_depth, args.num_nodes,
                    device, dropout=args.dropout, subgraph_size=args.subgraph_size,
                    node_dim=args.node_dim, dilation_exponential=args.dilation_exponential,
                    conv_channels=args.conv_channels, residual_channels=args.residual_channels,
                    skip_channels=args.skip_channels, end_channels= args.end_channels,
                    seq_length=args.seq_in_len, in_dim=args.in_dim, out_dim=args.seq_out_len,
                    layers=args.layers, propalpha=args.propalpha, tanhalpha=args.tanhalpha, layer_norm_affline=False)
    model = model.to(device)

    variances = [] # List to store the variance calculated in each epoch

    print(args)
    print('The recpetive field size is', model.receptive_field)
    nParams = sum([p.nelement() for p in model.parameters()])
    print('Number of model parameters is', nParams, flush=True)

    if args.L1Loss:
        criterion = nn.L1Loss(size_average=False).to(device)
    else:
        criterion = nn.MSELoss(size_average=False).to(device)
    evaluateL2 = nn.MSELoss(size_average=False).to(device)
    evaluateL1 = nn.L1Loss(size_average=False).to(device)


    best_val = 10000000
    optim = Optim(
        model.parameters(), args.optim, args.lr, args.clip, lr_decay=args.weight_decay
    )

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        print('begin training')
        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()
            train_loss = train(Data, Data.train[0], Data.train[1], model, criterion, optim, args.batch_size)
            val_loss, val_rae, val_corr, val_predict = evaluate(Data, Data.valid[0], Data.valid[1], model, evaluateL2, evaluateL1,
                                            args.batch_size)
            print(
                '| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.4f} | valid rse {:5.4f} | valid rae {:5.4f} | valid corr  {:5.4f}'.format(
                    epoch, (time.time() - epoch_start_time), train_loss, val_loss, val_rae, val_corr), flush=True)
            
            # Save the model if the validation loss is the best we've seen so far.
            if val_loss < best_val:
                with open(args.save, 'wb') as f:
                    torch.save(model, f)
                best_val = val_loss
            
            # Prediction's variance calculation during validation
            variance_val = np.var(val_predict)
            # print('variance_val: ', variance_val)
            variances.append(variance_val)

            # To save the adjacency matrix in a file
            if model.adjacency_matrix is not None:
                np.save("adjacency_matrix.npy", model.adjacency_matrix.cpu().detach().numpy())

            if epoch % 3 == 0: # Changed from 5 to 3
                # Monitoring purposes
                test_acc, test_rae, test_corr, test_predict = evaluate(Data, Data.test[0], Data.test[1], model, evaluateL2, evaluateL1,
                                                        args.batch_size)
                
                test_predict = test_predict * Data.scale.cpu().numpy() # Rescale from normalizing
                
                # Compute the confidence intervals for the predictions
                confidence_interval = np.percentile(test_predict, [5, 95], axis=0)
                interval_differences = confidence_interval[1] - confidence_interval[0]
                confidence_interval = np.vstack([confidence_interval, interval_differences])
                average_interval_size = np.mean(interval_differences)
                print("Average size for confidence intervals:", average_interval_size)

                # Save the results in a CSV file determined by the approach
                if args.sensitivity:
                    output_file = "predict_confinterv_result_sensitivity.csv"
                else:
                    output_file = "predict_confinterv_result.csv"
                output_format = '%.3f'
                np.savetxt(output_file, confidence_interval, delimiter=',', fmt=output_format)

                # Save the variances in a CSV file
                variances = np.array(variances)
                np.savetxt('variances.csv', variances, delimiter=',')
                
                print("test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}".format(test_acc, test_rae, test_corr), flush=True)

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    # Load the best saved model
    with open(args.save, 'rb') as f:
        model = torch.load(f)
    
    # Evaluate both sets with the best saved model
    vtest_acc, vtest_rae, vtest_corr, vtest_predict = evaluate(Data, Data.valid[0], Data.valid[1], model, evaluateL2, evaluateL1,
                                        args.batch_size)
    test_acc, test_rae, test_corr, test_predict = evaluate(Data, Data.test[0], Data.test[1], model, evaluateL2, evaluateL1,
                                        args.batch_size)
    print("final test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}".format(test_acc, test_rae, test_corr))
    return vtest_acc, vtest_rae, vtest_corr, test_acc, test_rae, test_corr

def main():

    if args.sensitivity:
        # Load the data
        file = open(args.data)
        data = np.loadtxt(file, delimiter=',')

        # Select a determined number of random clients to modify
        num_clients = data.shape[1]
        random_client_indices = np.random.choice(num_clients, args.sensitivity_num, replace=False) # Select 'args.sensitivity_num' random indices as the chosen clients

        for random_client_index in random_client_indices:
            random_client_data = data[:, random_client_index] # Select the corresponding column of the random client
            min_value = np.min(random_client_data)
            max_value = np.max(random_client_data)
            random_client_data = np.random.uniform(min_value, max_value, size=data.shape[0]) # Modify the column values with random values in a uniform distribution
            #print("Modified column: ", random_client_data)
            
            data[:, random_client_index] = random_client_data # Update the new column in 'data'

        print("Modified clients: ", random_client_indices)

        # Create the train, validation and test sets with the new data
        Data = DataLoaderS(args.data, 0.6, 0.2, device, args.horizon, args.seq_in_len, args.sensitivity, data, args.normalize)

    else:
        # Original 
        Data = DataLoaderS(args.data, 0.6, 0.2, device, args.horizon, args.seq_in_len, args.sensitivity, args.normalize)

    return main_training(Data)

        

if __name__ == "__main__":
    vacc = []
    vrae = []
    vcorr = []
    acc = []
    rae = []
    corr = []
    for i in range(args.exp_num): # Changed to allow different number of experiments
        val_acc, val_rae, val_corr, test_acc, test_rae, test_corr = main()
        vacc.append(val_acc)
        vrae.append(val_rae)
        vcorr.append(val_corr)
        acc.append(test_acc)
        rae.append(test_rae)
        corr.append(test_corr)
    print('\n\n')
    print(f'{args.exp_num} runs average') # Changed to allow different number of experiments
    print('\n\n')
    print("valid\trse\trae\tcorr")
    print("mean\t{:5.4f}\t{:5.4f}\t{:5.4f}".format(np.mean(vacc), np.mean(vrae), np.mean(vcorr)))
    print("std\t{:5.4f}\t{:5.4f}\t{:5.4f}".format(np.std(vacc), np.std(vrae), np.std(vcorr)))
    print('\n\n')
    print("test\trse\trae\tcorr")
    print("mean\t{:5.4f}\t{:5.4f}\t{:5.4f}".format(np.mean(acc), np.mean(rae), np.mean(corr)))
    print("std\t{:5.4f}\t{:5.4f}\t{:5.4f}".format(np.std(acc), np.std(rae), np.std(corr)))

