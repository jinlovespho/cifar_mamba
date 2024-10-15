from tqdm import tqdm
import torch 
from loss import model_forward, calculate_loss 
from torchvision.utils import save_image 
import wandb

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_one_epoch(epoch, model, train_loader, loss_fn, optim, scheduler, args):
    
    model.train()

    # for accuracy calculation
    tot_sample_num=0
    tot_correct_num=0

    tqdm_train = tqdm(train_loader, desc=f'Train Epoch: {epoch+1}/{args.tot_epoch}', unit='batch')
    for sample, target in tqdm_train:
        optim.zero_grad()

        # put datas to cuda 
        sample = sample.to(device)
        target = target.to(device)

        model_out = model_forward(model, sample)
        loss = loss_fn(model_out, target)
        loss.backward()
        optim.step()
        scheduler.step()

        # train loop acc 
        num_sample = len(sample)
        num_correct = (model_out.argmax(dim=-1)==target).float().sum()
        train_acc_loop = num_correct/num_sample*100

        train_loop_dic={
            'train_loss': loss.item(),
            'train_acc': train_acc_loop.item(),
            'lr': scheduler.get_last_lr()[0],
            'epoch': epoch+1
        }

        tqdm_train.set_postfix(train_loop_dic)
        if args.log_tool == 'wandb':
            wandb.log(train_loop_dic)

        tot_sample_num += num_sample
        tot_correct_num += num_correct
        

    train_acc = tot_correct_num / tot_sample_num
    print(f'Train Acc: {train_acc*100:.2f}')



