import os 
import argparse
import torch
import torch.nn as nn 
from torch.utils.data import DataLoader 
import wandb
import initialize
from train_one_epoch import train_one_epoch
from validate import val
import torchmetrics

def get_args():
    parser = argparse.ArgumentParser()
    
    # train data args 
    parser.add_argument('--train_ds', type=str)         # name of training dataset
    parser.add_argument('--train_ds_path', type=str)     # path to training dataset
    parser.add_argument('--train_height', type=int)
    parser.add_argument('--train_width', type=int)

    # val data args
    parser.add_argument('--val_ds', type=str)
    parser.add_argument('--val_ds_path', type=str)

    # train args 
    parser.add_argument('--lr', type=float) 
    parser.add_argument('--train_bs', type=int)     # training batch size
    parser.add_argument('--eval_bs', type=int)      # eval batch size 
    parser.add_argument('--tot_epoch', type=int)

    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--num_workers', type=int)
    

    # model args 
    parser.add_argument('--model', type=str)

    # eval args

    # save args 
    parser.add_argument('--log_tool', type=str)
    parser.add_argument('--wandb_proj_name', type=str)
    parser.add_argument('--wandb_exp_name', type=str)
    parser.add_argument('--save_path', type=str)

    # etc args 
    parser.add_argument('--measure_inf_time', action='store_true')

    return parser.parse_args()



if __name__ =='__main__':
    args = get_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # init dataset and loader 
    train_ds, val_ds = initialize.load_dataset(args)
    train_loader = DataLoader(dataset=train_ds, batch_size=args.train_bs, shuffle=True, num_workers=args.num_workers, drop_last=True, pin_memory=True)
    val_loader = DataLoader(dataset=val_ds, batch_size=args.eval_bs, shuffle=False, num_workers=args.num_workers, drop_last=True, pin_memory=True)

    # init model
    model = initialize.load_model(args)
    tot_params = sum(i.numel() for i in model.parameters())
    print(f'TOT PARAMS: {tot_params/1e6:.2f}M')

    # init optimizer, scheduler
    optim = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=39000)
    # scaler = torch.cuda.amp.GradScaler()

    # init loss function 
    loss_fn = nn.CrossEntropyLoss()
    # val_metric = torchmetrics.Accuracy(task="multiclass", num_classes=args.num_classes).to(device)

    # init log tool
    if args.log_tool == 'wandb':
        wandb.init( project = args.wandb_proj_name,
                    name = args.wandb_exp_name,
                    config = args,
                    dir=args.save_path)


    # training
    for epoch in range(args.tot_epoch):
        train_one_epoch(epoch, model, train_loader, loss_fn, optim, scheduler, args)
        val(epoch, model, val_loader, loss_fn, args)
        print('')

    print('FINISHED! Now Lets Eat Pho!!')



