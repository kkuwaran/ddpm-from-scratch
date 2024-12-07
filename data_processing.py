import multiprocessing
import random

from torch.utils.data import ConcatDataset, DataLoader, Subset
import torchvision



def get_dataloader(path: str, image_size: int, batch_size: int, num_samples: int = None):
    """Get dataloader for Stanford Cars dataset"""

    # define transform for dataset
    data_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((image_size, image_size)),
            # Flip horizontally with probability 0.5
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            # Scales data into [-1,1] 
            torchvision.transforms.Normalize(0.5, 0.5)
        ]
    )
    
    # get dataset
    train = torchvision.datasets.StanfordCars(root=path, download=False, 
                                         transform=data_transform, split='train')
    test = torchvision.datasets.StanfordCars(root=path, download=False, 
                                         transform=data_transform, split='test')
    dataset = ConcatDataset([train, test])

    total_samples = len(dataset)
    print(f"Size of the dataset: {total_samples}")

    # Sample from dataset
    if num_samples:
        random_indices = random.sample(range(total_samples), num_samples)
        dataset = Subset(dataset, random_indices)

    # get dataloader
    workers = multiprocessing.cpu_count()
    print(f"Using {workers} workers")
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=False, 
        pin_memory=True, persistent_workers=True if workers > 0 else False,
    )
    
    return dataloader