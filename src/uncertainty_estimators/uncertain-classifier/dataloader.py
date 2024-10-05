from torchvision import transforms, datasets
from data_augmentation import DataAugmentation

def get_dataset(dataset_name, augment_data, num_data_augmentations):
    if dataset_name == "ImageNet":
        train_dataset = datasets.ImageNet( # ImageNet or Imagenette
            # root = "~/datasets/ImageNet2012",   # BwUniCloud nette-download", #Net2012", # ImageNet2012",
            root = "/mnt/HDD1/datasets/ImageNet2012",   # local workstation
            split = "train",
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                DataAugmentation(augment_data, num_data_augmentations)
            ])
        )
        val_dataset = datasets.ImageNet( # ImageNet or Imagenette
            # root = "~/datasets/ImageNet2012",   # BwUniCloud nette-download", # Net2012", mnt/HDD1
            root = "/mnt/HDD1/datasets/ImageNet2012",   # local workstation
            split = "val",
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        )
        # remove duplicate labels
        class_labels = {i:i for i in range(0,1000)}
        image_size=(3, 224, 224)

    elif dataset_name == "Imagenette":
        train_dataset = datasets.Imagenette(
            root = "/mnt/HDD1/datasets/Imagenette",   # local workstation
            split = "train",
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                DataAugmentation(augment_data, num_data_augmentations)
            ])
            )
        val_dataset = datasets.Imagenette(
            root = "/mnt/HDD1/datasets/Imagenette",   # local workstation
            split = "val",
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            )
        class_labels = {'tench': 0, 'English springer': 1, 'cassette player': 2, 'chainsaw': 3, 'church': 4, 'French horn': 5, 'garbage truck': 6, 'gas pump': 7, 'golf ball': 8, 'parachute': 9}
        image_size = (3, 224, 224)
    
    elif dataset_name == "MNIST":
        train_dataset = datasets.MNIST(
            root = "/mnt/HDD1/datasets/mnist",   # local workstation
            train = True,
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1) ),
                DataAugmentation(augment_data, num_data_augmentations)
            ])
        )
        val_dataset = datasets.MNIST(
            root = "/mnt/HDD1/datasets/mnist",   # local workstation
            train = False,
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1) )
            ])
        )
        class_labels = train_dataset.class_to_idx
        image_size = (1, 28, 28)
    
    elif dataset_name == "GTSRB":
        train_dataset = datasets.GTSRB(
            root = "/mnt/HDD1/datasets/german_traffic_sign_recognition_benchmark",   # local workstation
            split = "train",
            transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.3403, 0.3121, 0.3214), std=(0.2724, 0.2608, 0.2669)),
                DataAugmentation(augment_data, num_data_augmentations)
            ])
        )
        val_dataset = datasets.GTSRB(
            root = "/mnt/HDD1/datasets/german_traffic_sign_recognition_benchmark",   # local workstation
            split = "test",
            transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.3403, 0.3121, 0.3214), std=(0.2724, 0.2608, 0.2669))
            ])
        )
        class_labels = {i:i for i in range(0,43)}
        image_size = (3, 32, 32)
    

    else:
        raise Exception("Dataset loader not implement for requested dataset: " + dataset_name) 

    return train_dataset, val_dataset, class_labels, image_size