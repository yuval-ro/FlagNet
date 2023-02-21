from torchvision import transforms

# Pre-Proccessing consts:
degrees_rotation = 30
size_crop = 224
size_resize = 256
normalize_mean = [.485, .456, .406]
normalize_std = [.229, .224, .225]

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
pilimg_transforms = transforms.Compose([transforms.Resize(size_resize),
                                      transforms.CenterCrop(size_crop),
                                      transforms.ToTensor()])
