import torch 
import torchvision 
import torchvision.transforms as transforms
from utils.autoaugment import CIFAR10Policy, SVHNPolicy, ImageNetPolicy, RandomCropPaste

def load_transform(args):
    train_transform = []
    val_transform = []
    
    if args.train_ds == 'cifar10' or args.train_ds == 'cifar100':
        train_transform += [transforms.RandomCrop(size=args.size, padding=args.padding),
                            transforms.RandomHorizontalFlip()]
        if args.autoaugment:
            train_transform += [CIFAR10Policy()]
        train_transform += [transforms.ToTensor(),
                            transforms.Normalize(mean=args.mean, std=args.std)]
        val_transform += [transforms.ToTensor(),
                          transforms.Normalize(mean=args.mean, std=args.std)]
        
    elif args.train_ds == 'imagenet':
        train_transform += [transforms.RandomResizedCrop(args.size),
                            transforms.RandomHorizontalFlip(),]
        if args.autoaugment:
            train_transform.append(ImageNetPolicy())
        train_transform += [transforms.ToTensor(),
                            transforms.Normalize(mean=args.mean, std=args.std)]  
        val_transform += [transforms.Resize(256),
                          transforms.CenterCrop(args.size),
                          transforms.ToTensor(),
                          transforms.Normalize(mean=args.mean, std=args.std)]
    
    train_transform = transforms.Compose(train_transform)
    val_transform = transforms.Compose(val_transform)
    return train_transform, val_transform


def load_dataset(args):
    ds_path = args.train_ds_path 

    if args.train_ds == 'cifar100':
        args.in_c = 3
        args.num_classes=100
        args.size = 32
        args.padding = 4
        args.mean, args.std = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
        args.autoaugment=True
        train_transform, val_transform = load_transform(args)
        train_ds = torchvision.datasets.CIFAR100(root=ds_path, download=True, train=True, transform=train_transform)
        val_ds = torchvision.datasets.CIFAR100(root=ds_path, download=True, train=False, transform=val_transform)
    
    elif args.train_ds == 'cifar10':
        args.in_c = 3
        args.num_classes=10
        args.size = 32
        args.padding = 4
        args.mean, args.std = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
        args.autoaugment=True
        train_transform, val_transform = load_transform(args)
        train_ds = torchvision.datasets.CIFAR10(root=ds_path, download=True, train=True, transform=train_transform)
        val_ds = torchvision.datasets.CIFAR10(root=ds_path, download=True, train=False, transform=val_transform)

    elif args.train_ds == 'imagenet1k':
        args.in_c = 3
        args.num_classes=1000
        args.size = 224
        args.mean, args.std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        train_transform, val_transform = load_transform(args)
        train_ds = torchvision.datasets.ImageNet()
        val_ds = torchvision.datasets.ImageNet()
    
    else:
        print('MISSING dataset! ')

    return train_ds, val_ds 
    

def load_model(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.model == 'mamba':
        from model.img_classifier import ImgClassifier
        
        block = 'mamba'
        n_layers = 6
        patch_size = 4
        img_size = 32
        embed_dim = 256
        dropout = 0.1
        n_layers = 6
        n_channels = 3
        n_class = args.num_classes

        model = ImgClassifier(patch_size, img_size, n_channels, embed_dim, n_layers, dropout, n_class)
        model = model.to(device)

    elif args.model == 'vit_small':
        pass 
    
    return model



