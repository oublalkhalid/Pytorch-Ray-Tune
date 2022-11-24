import torchvision.transforms as transforms
import torchvision.datasets as datasets

from torch.utils.data import DataLoader

# Training transforms.
def train_transforms():
    train_transform = transforms.Compose([
        transforms.Resize(224),
        # transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomVerticalFlip(p=0.5),
        # transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        # transforms.RandomRotation(degrees=(30, 70)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])
    return train_transform

# Validation transforms.
def valid_transforms():
    valid_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])
    return valid_transform

def get_datasets(DATA_ROOT_DIR):
    # Training dataset.
    train_dataset = datasets.ImageFolder(
        root=f"{DATA_ROOT_DIR}/TRAIN",
        transform=train_transforms()
    )
    # Validation dataset.
    valid_dataset = datasets.ImageFolder(
        root=f"{DATA_ROOT_DIR}/TEST_SIMPLE",
        transform=valid_transforms()
    )
    # Test dataset.
    test_dataset = datasets.ImageFolder(
        root=f"{DATA_ROOT_DIR}/TEST",
        transform=valid_transforms()
    )
    return (
        train_dataset, valid_dataset, 
        test_dataset, train_dataset.classes
    )

def get_data_loaders(
    train_dataset, valid_dataset, test_dataset,
    BATCH_SIZE, NUM_WORKERS
):
    # Training data loader.
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    # Validation data loader.
    valid_loader = DataLoader(
        valid_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    # Test data loader.
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    return train_loader, valid_loader, test_loader
