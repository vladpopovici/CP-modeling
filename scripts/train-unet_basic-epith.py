# export CUDA_VISIBLE_DEVICES="-1" for CPU-only

import tensorflow as tf
from cpm.data import PairedImageGenerator
from cpm.unet.basic import unet
from cpm.trainers import train_basic_generator

# Set some parameters
IMG_WIDTH    = 256
IMG_HEIGHT   = 256
IMG_CHANNELS = 3
NB_CLASSES   = 1

TRAIN_PATH = '/home/vlad/tmp/epith'
TEST_PATH = '/home/vlad/tmp/epith'
BATCH_SIZE = 128

with tf.device("cpu:0"):
    data_generator = PairedImageGenerator().flow_from_directory_segmentation(
        directory= TRAIN_PATH,
        image_subfolder='patch', mask_subfolder='mask',
        target_size={'width': IMG_WIDTH, 'height': IMG_HEIGHT},
        classes=NB_CLASSES,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    model = unet((IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))
    lg, model = train_basic_generator(model,
                                      train_data_generator=data_generator,
                                      valid_data_generator=data_generator,
                                      batch_size=2,
                                      epochs=5)