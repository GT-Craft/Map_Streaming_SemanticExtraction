from torch.utils.data import DataLoader
import torch
import albumentations as album
import utils.Massachusetts_utils as mutils

class Single_Dataset(torch.utils.data.Dataset):
    def __init__(self, img, mask, class_rgb_values = None, augmentation = None, preprocessing = None):
        self.img = img
        self.mask = mask
        self.class_rgb_values = class_rgb_values
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        image = self.img
        mask = self.mask
        mask = mutils.one_hot_encode(mask, self.class_rgb_values).astype(float)

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # image, mask convert ndarray to tensor
        # image = torch.from_numpy(image).float()
        # mask = torch.from_numpy(mask).float()

        return image, mask

    def __len__(self):
        return 1



def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn=None):
    """Construct preprocessing transform
    Args:
        preprocessing_fn (callable): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    """
    _transform = []
    if preprocessing_fn:
        _transform.append(album.Lambda(image=preprocessing_fn))
    _transform.append(album.Lambda(image=to_tensor, mask=to_tensor))

    return album.Compose(_transform)
