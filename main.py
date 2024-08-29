from os import write
import sys
import argparse
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from model.emgnn import EMGNN
import optuna

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')
    

class JsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(JsonEncoder, self).default(obj)

#log
class Logger(object):
    def __init__(self,fileN ="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN,"a")
 
    def write(self,message):
        self.terminal.write(message)
        self.log.write(message)
 
    def flush(self):
        pass



def evaluate(data, inputs, targets, model, evaluateL2, evaluateL1, batch_size):
    model.eval()
    total_loss = 0
    total_loss_l1 = 0
    n_samples = 0
    predict = None
    test = None

    for X, Y in data.get_batches(inputs, targets, batch_size, False):
        X = torch.unsqueeze(X,dim=1)
        X = X.transpose(2,3) #(B, F, N, T)
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

    rae = (total_loss_l1 / n_samples) / data.rae

    scale = data.scale.expand(predict.size(0), data.m)
    predict = predict * scale
    test = test * scale
    predict = predict.data.cpu().numpy()
    Ytest = test.data.cpu().numpy()
    predict = predict[:,0]
    Ytest = Ytest[:,0]
    print("predict:")
    print(predict.shape)
    print("Ytest")
    print(Ytest.shape) 
    scores = get_scores(predict, Ytest,0, 'single')
    scores['RAE'] = rae.item()     
    return scores, predict, Ytest


def train(data, X, Y, model, criterion, optim, batch_size):
    model.train()
    total_loss = 0   
    n_samples = 0
    iter = 0
    for tx, ty in data.get_batches(X, Y, batch_size, True):
        model.zero_grad()
        tx = torch.unsqueeze(tx,dim=1)
        tx = tx.transpose(2,3)
                     
        output = model(tx)
        output = torch.squeeze(output)
        scale = data.scale.expand(output.size(0), data.m)
        loss = criterion(output * scale, ty * scale)                  
        loss.backward()
        total_loss += loss.item()   
        n_samples += (output.size(0) * data.m)
        grad_norm = optim.step()

        if iter%100==0:
            print('iter:{:3d} | loss: {:.8f}'.format(iter,loss.item()/(output.size(0) * data.m)))
        iter += 1
    return total_loss / n_samples


parser = argparse.ArgumentParser(description='PyTorch forecasting')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument('--device',type=str,default=device,help='')
#parser.add_argument('--device',type=str,default='cuda:0',help='')
parser.add_argument('--data', type=str, default='data_allh5', help='the name of the dataset')#must use h5 type data
parser.add_argument('--horizon', type=int, default=1)
parser.add_argument('--expid', type=str, default='1',
                    help='The folder name used to save model, output and evaluation metrics. This can be set to any word')
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--L1Loss', type=bool, default=True)
parser.add_argument('--normalize', type=int, default=2)

parser.add_argument('--dynamic_embedding',type=int,default=16, help='the dimension of dynamic node representation')
parser.add_argument('--dynamic_interval',type=list, default=[31,21,14,5,1],help='time intervals for each layer')
parser.add_argument('--nodes_num',type=int,default=11,help='number of nodes/variables')
parser.add_argument('--in_len',type=int,default=168,help='input sequence length')
parser.add_argument('--out_len',type=int,default=1,help='output sequence length')
parser.add_argument('--in_dim',type=int,default=1,help='inputs dimension')
parser.add_argument('--out_dim',type=int,default=1,help='outputs dimension')
parser.add_argument('--layers',type=int,default=5,help='number of layers')
parser.add_argument('--gcn_depth',type=int,default=2,help='graph convolution depth')
parser.add_argument('--conv_ch',type=int,default=8,help='convolution channels')
parser.add_argument('--res_ch',type=int,default=8,help='residual channels')
parser.add_argument('--skip_ch',type=int,default=16,help='skip channels')
parser.add_argument('--end_ch',type=int,default=32,help='end channels')
parser.add_argument('--kernel_set',type=list,default=[2,3,6,7],help='the kernel set in TCN')
parser.add_argument('--dilation_exp',type=int,default=2,help='dilation exponential')
parser.add_argument('--static_embedding',type=int,default=40,help='the dimension of static node representation')
parser.add_argument('--propalpha',type=float,default=0.05,help='prop alpha')
parser.add_argument('--lr',type=float,default=0.0001,help='learning rate')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')

parser.add_argument('--batch_size',type=int,default=4, help='batch size')
parser.add_argument('--weight_decay',type=float,default=0.00001,help='weight decay rate')
parser.add_argument('--clip',type=int,default=5,help='clip')
parser.add_argument('--epochs',type=int,default=50,help='')
parser.add_argument('--early_stop',type=str_to_bool,default=False,help='')
parser.add_argument('--early_stop_steps',type=int,default=15,help='')
parser.add_argument('--runs',type=int,default=1,help='number of runs')
parser.add_argument('--optuna_trials', type=int, default=None, help='Number of Optuna trials for hyperparameter tuning. If None, Optuna is not used.')

args = parser.parse_args()
device = torch.device(args.device)
torch.set_num_threads(3)

def objective(trail):
    #params for optuna tunning
    gcn_depth = trail.suggest_int('gcn_depth', 1, 6)
    dropout = trail.suggest_float('dropout', 0.1, 0.2, 0.3)
    lr = trail.suggest_float('lr',0.1, 0.01, 0.001, 0.0001)

    model = EMGNN(args.dynamic_embedding, args.dynamic_interval, args.nodes_num, args.in_len, args.out_len, args.in_dim, args.out_dim, 1, args.layers,
                  conv_ch=args.conv_ch, res_ch=args.res_ch,
                  skip_ch=args.skip_ch, end_ch= args.end_ch, kernel_set=args.kernel_set,
                  dilation_exp=args.dilation_exp, gcn_depth=gcn_depth,
                  device=device, fc_dim = (node_fea.shape[0]-18)*16, static_embedding=args.static_embedding,                  
                  dropout=dropout,  propalpha=args.propalpha, layer_norm_affline=False,
                  static_feat = node_fea)

    model = model.to(device)

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)

    # Define loss functions
    criterion = nn.L1Loss(size_average=False).to(device) if args.L1Loss else nn.MSELoss(size_average=False).to(device)
    evaluateL2 = nn.MSELoss(size_average=False).to(device)
    evaluateL1 = nn.L1Loss(size_average=False).to(device)

    # Training loop
    best_val_loss = 10000000
    best_epoch = 10000

    for epoch in range(1, args.epochs + 1, 1):
        train_loss= train(Data, Data.train[0], Data.train[1], model, criterion, optim, args.batch_size)
        val_scores, P, Y = evaluate(Data, Data.valid[0], Data.valid[1], model, evaluateL2, evaluateL1,
                                           args.batch_size)
        val_loss = val_scores['RSE']

        # Prune trials
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        return best_val_loss

def main(runid):
    save_folder = os.path.join('saves', args.data, args.expid, 'horizon_'+str(args.horizon), str(runid))
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    model_path = os.path.join(save_folder,'best-model.pt')

    sys.stdout = Logger(os.path.join(save_folder,'log.txt')) 

    Data = DataLoaderS(args.data, 0.7, 0.15, device, args.horizon, args.in_len, args.normalize) 
    
    node_fea = get_node_fea(args.data, 0.7)
    node_fea = torch.tensor(node_fea).type(torch.FloatTensor).to(args.device)

    if args.optuna_trials:
        study=optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=args.optuna_trials)

       # Print best hyperparameters
        print('Best trial:')
        trial = study.best_trial

        print('  Value: {}'.format(trial.value))
        print('  Params: ')
        for key, value in trial.params.items():
            print('    {}: {}'.format(key, value))

        # Use the best hyperparameters to train and evaluate the final model
        best_params = trial.params
    else:
        best_params = {
        'gcn_depth': args.gcn_depth,
        'lr': args.lr,
        'dropout': args.dropout
        }

    model = EMGNN(args.dynamic_embedding, args.dynamic_interval, args.nodes_num, args.in_len, args.out_len, args.in_dim, args.out_dim, 1, args.layers,#标记过红点
                  conv_ch=args.conv_ch, res_ch=args.res_ch,
                  skip_ch=args.skip_ch, end_ch= args.end_ch, kernel_set=args.kernel_set,
                  dilation_exp=args.dilation_exp, gcn_depth=best_params['gcn_depth'],
                  device=device, fc_dim = (node_fea.shape[0]-18)*16, static_embedding=args.static_embedding,                  
                  dropout=best_params['dropout'],  propalpha=args.propalpha, layer_norm_affline=False,
                  static_feat = node_fea)

    model = model.to(device)

    print(args)
    print('The recpetive field size is', model.receptive_field)
    nParams = sum([p.nelement() for p in model.parameters()])
    print('Number of model parameters is', nParams, flush=True)
    print(model)

    run_folder = os.path.join(save_folder,'run')
    if not os.path.exists(run_folder):
        os.makedirs(run_folder)
    writer = SummaryWriter(run_folder)

    if args.L1Loss:
        criterion = nn.L1Loss(size_average=False).to(device)
    else:
        criterion = nn.MSELoss(size_average=False).to(device)
    evaluateL2 = nn.MSELoss(size_average=False).to(device)
    evaluateL1 = nn.L1Loss(size_average=False).to(device)
    

    best_val = 10000000
    best_epoch = 10000
    optim = Optim(
        model.parameters(), args.optim, best_params['lr'], args.clip, lr_decay=args.weight_decay
    )
        
    try:
        print('begin training')
        for epoch in range(1, args.epochs + 1, 1):
            epoch_start_time = time.time()
            train_loss= train(Data, Data.train[0], Data.train[1], model, criterion, optim, args.batch_size)
            writer.add_scalars('train_loss', {'train':train_loss}, epoch)
            val_scores, P, Y = evaluate(Data, Data.valid[0], Data.valid[1], model, evaluateL2, evaluateL1,
                                               args.batch_size)
            print(
                '| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.4f} | valid rse {:5.4f} | valid rae {:5.4f} | valid corr  {:5.4f}'.format(
                    epoch, (time.time() - epoch_start_time), train_loss, val_scores['RSE'], val_scores['RAE'], val_scores['CORR']), flush=True)
            writer.add_scalars('rse',{'valid':val_scores['RSE']},global_step = epoch)
            writer.add_scalars('corr',{'valid':val_scores['CORR']},global_step = epoch)
           
            # Save the model if the validation loss is the best we've seen so far.
            val_loss = val_scores['RSE']
            if val_loss < best_val:
                print('save the model at epoch ' + str(epoch)+'*********')
                with open( model_path, 'wb') as f:
                    torch.save(model, f)
                best_val = val_loss
                best_epoch = epoch
            elif args.early_stop and  epoch - best_epoch > args.early_stop_steps:
                print('best epoch:', best_epoch)
                raise ValueError('Early stopped.')
            if epoch % 5 == 0:
                test_scores, P, Y = evaluate(Data, Data.test[0], Data.test[1], model, evaluateL2, evaluateL1,
                                                args.batch_size)
                print("test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}".format( test_scores['RSE'], test_scores['RAE'], test_scores['CORR']), flush=True)
                writer.add_scalars('rse',{'test':test_scores['RSE']},global_step = epoch)
                writer.add_scalars('corr',{'test':test_scores['CORR']},global_step = epoch)
    
    except (ValueError, KeyboardInterrupt) as e:
        print(e)
        # except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')
    
    print('best epoch:', best_epoch)   
    
    # Load the best saved model.
    with open(model_path, 'rb') as f:
        model = torch.load(f)

    val_scores, P, Y = evaluate(Data, Data.valid[0], Data.valid[1], model, evaluateL2, evaluateL1,
                                         args.batch_size)
    test_scores, predict, Ytest = evaluate(Data, Data.test[0], Data.test[1], model, evaluateL2, evaluateL1,
                                         args.batch_size)
    
    # save test results
    np.savez(os.path.join(save_folder,'test-results.npz'), predictions=predict, targets=Ytest)
    print(json.dumps(test_scores, cls=JsonEncoder, indent=4))
   
    with open(os.path.join(save_folder,'test-scores.json'), 'w+') as f:
        json.dump(test_scores, f, cls=JsonEncoder, indent=4)
    
    #print("final test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}".format(test_scores['RSE'], test_scores['RAE'], test_scores['CORR']))
    return val_scores['RSE'], val_scores['RAE'], val_scores['CORR'], test_scores['RSE'], test_scores['RAE'], test_scores['CORR']
   
    
if __name__ == "__main__":
    vacc = []
    vrae = []
    vcorr = []
    acc = []
    rae = []
    corr = []
    for i in range(args.runs):
        val_acc, val_rae, val_corr, test_acc, test_rae, test_corr = main(i)
        vacc.append(val_acc)
        vrae.append(val_rae)
        vcorr.append(val_corr)
        acc.append(test_acc)
        rae.append(test_rae)
        corr.append(test_corr)
    print('\n\n')
    print(str(args.runs)+' runs average')
    print('\n\n')
    print("valid\trse\trae\tcorr")
    print("mean\t{:5.4f}\t{:5.4f}\t{:5.4f}".format(np.mean(vacc), np.mean(vrae), np.mean(vcorr)))
    print("std\t{:5.4f}\t{:5.4f}\t{:5.4f}".format(np.std(vacc), np.std(vrae), np.std(vcorr)))
    print('\n\n')
    print("test\trse\trae\tcorr")
    print("mean\t{:5.4f}\t{:5.4f}\t{:5.4f}".format(np.mean(acc), np.mean(rae), np.mean(corr)))
    print("std\t{:5.4f}\t{:5.4f}\t{:5.4f}".format(np.std(acc), np.std(rae), np.std(corr)))

