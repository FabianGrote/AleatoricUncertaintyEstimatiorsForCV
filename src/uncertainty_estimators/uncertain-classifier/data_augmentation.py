# Code is original from tensorflow version of https://github.com/berenslab/ttaug-midl2018/blob/master/TTAUG_MIDL2018/utils/DataAugmentation.py
import torch
from torchvision.transforms import InterpolationMode, ColorJitter, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, RandomResizedCrop

import random

class DataAugmentation(object):
    """ Apply multiple transformations on the input data as presented in
        "Test-time Data Augmentation for Estimation of Heteroscedastic 
        Aleatoric Uncertainty in Deep Neural Networks """

    def __init__(self, augment_data, num_data_augmentations, rotation_and_flip, val):
        self.augment_data = augment_data
        self.num_data_augmentations = num_data_augmentations
        self.rotation_and_flip = rotation_and_flip
        self.val = val
    
    def __call__(self, sample):
        if self.augment_data:
            augmented_sample = self.data_augmentation(sample)
            return augmented_sample
        else:
            return sample

    def do_nothing(self, inputs):
        return inputs


    # crops and resizes and image with possibility p=0.5
    def crop_and_resize(self, image):
        # create random uniform value between 0 and 1
        rand = random.uniform(0., 1.)

        if rand >= 0.5:
            crop_resized_image = RandomResizedCrop(
                size=(image.shape[-2], image.shape[-1]), 
                scale=(0.33, 1.0), 
                ratio=(1.0, 1.0), 
                interpolation=InterpolationMode.BILINEAR
            )(image)
            return crop_resized_image
        else:
            return image


    def data_augmentation(self, inputs):
         ############  data augmentation on the fly  ###################
        output_list = []
        if self.val:
            inputs = torch.repeat_interleave(inputs.unsqueeze(dim=0), self.num_data_augmentations, dim=0)
    
        for image in inputs:
            image = image.unsqueeze(dim=0)
            #  PART 0: First, randomly crop images. Randomness has two folds:
            #  i) Coin toss: To crop or not to crop ii) Bounding box corners are randomly generated
            #  This part can considered as an extension to the sampling process.
            #  random crops from images resized to original shape (zooming effect)
            image = self.crop_and_resize(image)

            # PART 1: Manipulation of pixels values via brightness, hue, saturation, and contrast adjustments

            # Randomly change the brightness, contrast, saturation and hue of an image. If the image is torch Tensor, 
            # it is expected to have […, 1 or 3, H, W] shape, where … means an arbitrary number of leading dimensions.
            image = ColorJitter(
                # random brightness adjustment sampled from [max(0, 1 - brightness), 1 + brightness] 
                # 0 is black image, 1 is original image, 2 is double brightness.
                brightness=0.5,
                contrast=(0., 3.0), 
                #  How much to jitter saturation. saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation] or the given [min, max]
                saturation=(0., 3.0), 
                # random hue adjustments sampled from [-hue, hue]. hue must be in [0, 0.5].
                hue=0.5
            )(image)

            # make sure that pixel values are in [0., 1.]
            # inputs = torch.minimum(inputs, 1.0)
            # inputs = torch.maximum(inputs, 0.0)
            
            if self.rotation_and_flip:
                # PART 2: Physical transformations on images: Flip LR, Flip UD, Rotate

                # randomly mirror images horizontally
                image = RandomHorizontalFlip(p=0.5)(image)

                # randomly mirror images vertically
                image = RandomVerticalFlip(p=0.5)(image)

                # random translations
                # inputs = tf.contrib.image.translate(inputs,
                #                                    translations=tf.random_uniform(shape=[tf.shape(inputs)[0], 2],
                #                                                                   minval=-50, maxval=50, dtype=tf.float32
                #                                                                   ),
                #                                    interpolation='NEAREST',
                #                                    name=None
                #                                    )

                # random rotations
                image = RandomRotation(degrees=(0., 360.), interpolation=InterpolationMode.NEAREST)(image)
                
            image = image.squeeze(dim=0)
            output_list.append(image)
        
        output = torch.stack(output_list)

        return output

