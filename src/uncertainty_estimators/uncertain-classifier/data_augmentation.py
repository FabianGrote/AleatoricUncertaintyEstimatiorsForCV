# Code is original from tensorflow version of https://github.com/berenslab/ttaug-midl2018/blob/master/TTAUG_MIDL2018/utils/DataAugmentation.py
import torch
from torchvision.transforms import InterpolationMode, ColorJitter, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, RandomResizedCrop

import random

class DataAugmentation(object):
    """ Apply multiple transformations on the input data as presented in
        "Test-time Data Augmentation for Estimation of Heteroscedastic 
        Aleatoric Uncertainty in Deep Neural Networks """

    def __init__(self, augment_data):
        self.augment_data = augment_data

    def __call__(self, sample):
        if self.augment_data:
            augmented_sample = self.data_augmentation(sample)
            # inputs = torch.cond(
            #     torch.squeeze(torch.greater_equal(torch.random_uniform(shape=[1], minval=0., maxval=1.0, dtype=torch.float32),
            #                             torch.constant(value=0.5, dtype=torch.float32)
            #                             )
            #         ),
            #     true_fn=lambda: self.data_augmentation(inputs),
            #     false_fn=lambda: self.do_nothing(inputs)
            # )
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
        #  PART 0: First, randomly crop images. Randomness has two folds:
        #  i) Coin toss: To crop or not to crop ii) Bounding box corners are randomly generated
        #  This part can considered as an extension to the sampling process.
        #  random crops from images resized to original shape (zooming effect)
        inputs = self.crop_and_resize(inputs)

        # PART 1: Manipulation of pixels values via brightness, hue, saturation, and contrast adjustments

        # Randomly change the brightness, contrast, saturation and hue of an image. If the image is torch Tensor, 
        # it is expected to have […, 1 or 3, H, W] shape, where … means an arbitrary number of leading dimensions.
        inputs = ColorJitter(
            # random brightness adjustment sampled from [max(0, 1 - brightness), 1 + brightness] 
            # 0 is black image, 1 is original image, 2 is double brightness.
            brightness=0.5,
            contrast=(0., 3.0), 
            #  How much to jitter saturation. saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation] or the given [min, max]
            saturation=(0., 3.0), 
            # random hue adjustments sampled from [-hue, hue]. hue must be in [0, 0.5].
            hue=0.5
        )(inputs)

        # make sure that pixel values are in [0., 1.]
        #inputs = torch.minimum(inputs, 1.0)
        #inputs = torch.maximum(inputs, 0.0)

        # PART 2: Physical transformations on images: Flip LR, Flip UD, Rotate

        # randomly mirror images horizontally
        inputs = RandomHorizontalFlip(p=0.5)(inputs)

        # randomly mirror images vertically
        inputs = RandomVerticalFlip(p=0.5)(inputs)

        # random translations
        #inputs = tf.contrib.image.translate(inputs,
        #                                    translations=tf.random_uniform(shape=[tf.shape(inputs)[0], 2],
        #                                                                   minval=-50, maxval=50, dtype=tf.float32
        #                                                                   ),
        #                                    interpolation='NEAREST',
        #                                    name=None
        #                                    )

        # random rotations
        inputs = RandomRotation(degrees=(0., 360.), interpolation=InterpolationMode.NEAREST)(inputs)

        return inputs

