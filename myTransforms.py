from torchvision import transforms

# Pre-Proccessing consts:
degrees_rotation = 30
size_crop = 224
size_resize = 256
normalize_mean = [0.485, 0.456, 0.406]
normalize_std = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([transforms.RandomRotation(degrees_rotation),
                                       transforms.RandomResizedCrop(size_crop),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize(normalize_mean, normalize_std)])
valid_transforms = transforms.Compose([transforms.Resize(size_resize), 
                                       transforms.CenterCrop(size_crop),
                                       transforms.ToTensor(),
                                       transforms.Normalize(normalize_mean, normalize_std)])
test_transforms = transforms.Compose([transforms.Resize(size_resize),
                                      transforms.CenterCrop(size_crop),
                                      transforms.ToTensor(),
                                      transforms.Normalize(normalize_mean, normalize_std)])
