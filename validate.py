from tqdm import tqdm
import torch 
from loss import model_forward, model_forward_inf_time, calculate_loss
import wandb

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def val(epoch, model, val_loader, loss_fn, args):

    # for acc calculation
    tot_sample_num=0
    tot_correct_num=0
    tot_fwd_pass_time=0

    with torch.no_grad():
        model.eval()

        tqdm_val = tqdm(val_loader, desc=f'Val Epoch: {epoch+1}/{args.tot_epoch}', unit='batch')
        for sample, target in tqdm_val:

            sample = sample.to(device)
            target = target.to(device)

            if args.measure_inf_time:
                model_out, inf_time = model_forward_inf_time(model, sample)     # inf_time = throughput 으로 정의
                tot_fwd_pass_time += inf_time
            else:
                model_out = model_forward(model, sample)

            loss = loss_fn(model_out, target)
            
            # val loop acc 
            num_sample = len(sample)
            num_correct = (model_out.argmax(dim=-1)==target).float().sum()

            tot_sample_num += num_sample
            tot_correct_num += num_correct
            # print(f'val loss: {loss:.2f}')

        val_acc = tot_correct_num / tot_sample_num
        print(f'Val Acc: {val_acc*100:.2f}')
        if args.log_tool == 'wandb':
            wandb.log({'One Epoch Val Acc': round(val_acc.item()*100, 2)})

        if args.measure_inf_time:
            ''' Latency = 1배치를 처리하는데 걸리는 시간 = s/b
                Throughput = 1초당 처리할 수 있는 배치 수 = b/s
                쉽게 말해 둘은 역수 관계인듯.
            '''
            throughput = tot_sample_num / tot_fwd_pass_time    # img/ms
            throughput = throughput * 1000  # img/s
            print(f'Throughput: {throughput:.1f} [img/s]')
            if args.log_tool == 'wandb':
                wandb.log({'Throughput [img/s]':round(throughput,1)})