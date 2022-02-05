from dataloaders.datasets import samsung_SEM, samsung_SEM_BE, samsung_CAD_BE
from torch.utils.data import DataLoader
from mypath import Path

def make_data_loader(args, **kwargs):


    train_set = samsung_CAD_BE.CADSegmentation(args, split='train', base_dir=args.dataset,)
    #val_set = samsung_CAD_BE.CADSegmentation(args, split='val', base_dir=Path.db_root_dir(args.dataset),)
    test_set = samsung_CAD_BE.CADSegmentation(args, split='test', base_dir=args.dataset,)

    num_class = train_set.NUM_CLASSES
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    #val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

    return train_loader, test_loader, test_loader, num_class





