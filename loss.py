import torch 

def model_forward(model, sample):    

    # train
    if model.training:
        model_out = model(sample)
    # val
    else:
        model_out = model(sample)
    
    return model_out


def model_forward_inf_time(model, sample):

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    starter.record()
    model_out = model(sample)
    ender.record()
    # WAIT FOR GPU SYNC
    torch.cuda.synchronize()
    batch_inf_time = starter.elapsed_time(ender)  # time(ms) spend inferencing one batch

    return model_out, batch_inf_time


def calculate_loss(loss_fn, model_out, target):
    loss = loss_fn(model_out, target)
    return loss