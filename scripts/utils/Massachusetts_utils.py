import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import albumentations as album
import torch
import cv2


class Massachusetts_Dataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, mask_dir, class_rgb_values = None, augmentation = None, preprocessing = None):
        self.img_paths = [os.path.join(img_dir, img_id) for img_id in sorted(os.listdir(img_dir))]
        self.mask_paths = [os.path.join(mask_dir, mask_id) for mask_id in sorted(os.listdir(mask_dir))]
        self.class_rgb_values = class_rgb_values
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        image = cv2.cvtColor(cv2.imread(self.img_paths[i]), cv2.COLOR_BGR2RGB)

        mask = cv2.cvtColor(cv2.imread(self.mask_paths[i]), cv2.COLOR_BGR2RGB)
        mask = one_hot_encode(mask, self.class_rgb_values).astype(float)

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.img_paths)


import segmentation_models_pytorch as smp
class Ms_Map_Processor():
    def __init__(self, encoder='resnet50', encoder_weights='imagenet'):
        preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, encoder_weights)
        self.preprocessing = get_preprocessing(preprocessing_fn)
        self.pad = get_validation_augmentation()

    def process_map_patch(self, patch_file, class_rgb_values, do_contrast=False):
        image = cv2.imread(patch_file)
        if do_contrast is False:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            lab= cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_channel, a, b = cv2.split(lab) # converting to LAB color space
            # Applying CLAHE to L-channel
            # feel free to try different values for the limit and grid size:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            cl = clahe.apply(l_channel)
            # merge the CLAHE enhanced L-channel with the a and b channel
            limg = cv2.merge((cl,a,b))
            # Converting image from LAB Color model to BGR color spcae
            image = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB) # BGR2RGB

        mask = one_hot_encode(image, class_rgb_values).astype(float)

        padded = self.pad(image=image, mask=mask)
        padded_image, padded_mask = padded['image'], padded['mask']

        preprocessed = self.preprocessing(image=padded_image, mask=padded_mask)
        preprocessed_image, preprocessed_mask = preprocessed['image'], preprocessed['mask']

        return image, preprocessed_image, preprocessed_mask


class Massachusetts_Helper:
    x_train_dir = ""
    y_train_dir = ""

    x_valid_dir = ""
    y_valid_dir = ""

    x_test_dir = ""
    y_test_dir = ""

    class_names = []
    class_rgb_values = []

    def initialize(self, data_dir):
        self.x_train_dir = os.path.join(data_dir, 'tiff/train')
        self.y_train_dir = os.path.join(data_dir, 'tiff/train_labels')

        self.x_valid_dir = os.path.join(data_dir, 'tiff/val')
        self.y_valid_dir = os.path.join(data_dir, 'tiff/val_labels')

        self.x_test_dir = os.path.join(data_dir, 'tiff/test')
        self.y_test_dir = os.path.join(data_dir, 'tiff/test_labels')

        class_dict = pd.read_csv(os.path.join(data_dir, 'label_class_dict.csv'))
        self.class_names = class_dict['name'].tolist()
        self.class_rgb_values = class_dict[['r', 'g', 'b']].values.tolist()

        print(f'{data_dir} classes and RGB values in labels:')
        print('Class Names: ', self.class_names)
        print('Class RGB values: ', self.class_rgb_values)

    def select_classes(self, classes=[]):
        select_class_indices = [self.class_names.index(cls.lower()) for cls in classes]
        select_class_rgb_values = np.array(self.class_rgb_values)[select_class_indices]
        return select_class_rgb_values



def visualize(**images):
    n_images = len(images)
    plt.figure(figsize=(20, 8))
    for idx, (name, image) in enumerate(images.items()):
        plt.subplot(1, n_images, idx+1)
        plt.xticks([])
        plt.yticks([])
        plt.title(name.replace('_', ' ').title(), fontsize=20)
        plt.imshow(image)
    plt.show()


def visualize_single(image):
    plt.figure(figsize=(1, 1))
    plt.xticks([])
    plt.yticks([])
    plt.imshow(image)
    plt.show()


def one_hot_encode(label, label_values):
    semantic_map = []
    for colour in label_values:
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)
    return semantic_map


def reverse_one_hot(image):
    x = np.argmax(image, axis=-1)
    return x


def colour_code_segmentation(image, label_values):
    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]
    return x


def get_training_augmentation():
    train_transform = [
        album.RandomCrop(height=256, width=256, always_apply=True),
        album.OneOf([album.HorizontalFlip(p=1),
                     album.VerticalFlip(p=1),
                     album.RandomRotate90(p=1),], p=0.75),
    ]
    return album.Compose(train_transform)


def get_validation_augmentation():
    validation_transform = [
        album.PadIfNeeded(min_height=1536, min_width=1536, always_apply=True, border_mode=0),
    ]
    return album.Compose(validation_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def to_tensor_basic(x):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn=None):
    _transform = []
    if preprocessing_fn:
        _transform.append(album.Lambda(image=preprocessing_fn))
    _transform.append(album.Lambda(image=to_tensor, mask=to_tensor))
    return album.Compose(_transform)


def crop_img(image, target_image_dims=[1500,1500,3]):
    target_size = target_image_dims[0]
    image_size = len(image)
    padding = (image_size - target_size) // 2

    if padding<0:
        return image

    return image[
        padding:image_size - padding,
        padding:image_size - padding,
        :,
    ]
