#
# DATA: data loaders
#

import tensorflow as tf
import tensorflow.keras.preprocessing.image as kpi
import tensorflow.keras.backend as K

import numpy as np
import PIL
from typing import Optional, Union
from pathlib import Path



def _list_valid_filenames_in_directory_segmentation(directory: Union[Path,str],
                                                    image_subfolder: str,
                                                    mask_subfolder: str,
                                                    white_list_formats: str) -> tuple[int, list, list]:

    """Count files with extension in `white_list_formats` contained in a directory.

    # Arguments
        directory: absolute path to the directory containing files to be counted
        white_list_formats: set of strings containing allowed extensions for
            the files to be counted.

    # Returns
        the count of files with extension in `white_list_formats` contained in
        the directory.
    """
    image_filenames = []
    mask_filenames = []
    directory = Path(directory)
    image_directory = directory / image_subfolder
    mask_directory = directory / mask_subfolder

    for img_path in image_directory.rglob("**/*"):
        if img_path.suffix.lower() not in white_list_formats:
            continue
        # find the corresponding mask:
        mask_path = mask_directory / img_path.relative_to(image_directory)
        if not mask_path.exists() or not mask_path.is_file():
            continue
        # image and its mask are present, add them to lists
        image_filenames.append(img_path)
        mask_filenames.append(mask_path)

    return len(image_filenames), image_filenames, mask_filenames
##-


def load_mask(path: Path, target_size: Optional[dict]=None) -> PIL.Image:
    """Loads an image into PIL format.

    # Arguments
        path: Path to image file
        grayscale: Boolean, whether to load the image as grayscale.
        target_size: Either `None` (defaults to original size)
            or dict{img_height:..., img_width:...}`.

    # Returns
        A PIL Image instance.

    # Raises
        ImportError: if PIL is not available.
    """
    img = PIL.Image.open(str(path))

    if target_size:
        if img.width != target_size['width'] or \
                img.height != target_size['height']:
            img = img.resize((target_size['width'], target_size['height']),
                             resample=PIL.Image.BILINEAR)
    return img
##-


class PairedImageGenerator(kpi.ImageDataGenerator):
    """Structure of the image databse:
    "directory"
    +-- "image_subfolder"
    |   +-- image1.png
    |   +-- image2.png
    |   +-- ...
    +-- "mask_subfolder"
        +-- image1.png
        +-- image2.png
        +-- ...
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def convert_to_onehot(self, x: np.array, classes: int, data_format: str='channel_last') -> np.array:

        if np.amax(x) > classes:
            raise AssertionError('Mask data does not match declared number of classes.')

        x = np.squeeze(x)
        x_ = (np.arange(classes) == x[..., None]).astype(dtype=K.floatx())

        if data_format == 'channels_first':
            x_ = np.rollaxis(x_,2,0)

        return x_


    def flow_from_directory_segmentation(self, directory: str,image_subfolder: str, mask_subfolder: str,
                                         target_size: dict[str,int]={'width':256, 'height':256},
                                         color_mode: str='rgb',
                                         batch_size: int=32, shuffle: bool=True, seed: Optional[int]=None,
                                         save_to_dir: Optional[str]=None, classes: Optional[int]=None,
                                         save_prefix: str='', save_format: str='png', follow_links: bool=False):

        """Iterator capable of reading images from a directory on disk.

        Args
            directory: Path to the directory to read images from.
            image_subfolder: folder name containing images. # new
            mask_subfolder: folder name containing masks. # new
            target_size: tuple of integers, dimensions to resize input images to.
            color_mode: One of `"rgb"`, `"grayscale"`. Color mode to read images.
            batch_size: Integer, size of a batch.
            shuffle: Boolean, whether to shuffle the data between epochs.
            seed: Random seed for data shuffling.
            save_to_dir: Optional directory where to save the pictures being yielded, in a viewable format. This is
                useful for visualizing the random transformations being applied, for debugging purposes.
            save_prefix: String prefix to use for saving sample images (if `save_to_dir` is set).
            save_format: Format to use for saving sample images (if `save_to_dir` is set).
            follow_links: Whether to follow symlinks inside class subdirectories (default: False).

        """

        return DirectoryIteratorSegmentation(directory,
                                             self,
                                             target_size=target_size, color_mode=color_mode,
                                             classes=classes,
                                             data_format=self.data_format,
                                             batch_size=batch_size, shuffle=shuffle, seed=seed,
                                             save_to_dir=save_to_dir,
                                             save_prefix=save_prefix,
                                             save_format=save_format,
                                             follow_links=follow_links,
                                             image_subfolder=image_subfolder,
                                             mask_subfolder=mask_subfolder
                                             )

##-
class DirectoryIteratorSegmentation(kpi.Iterator):

    def __init__(self, directory: str, image_data_generator: kpi.ImageDataGenerator,
                 image_subfolder: str, mask_subfolder: str, classes: int,
                 target_size: dict[str, int]={'width':256, 'height':256},
                 color_mode: str='rgb',
                 batch_size: int=32, shuffle: bool=True, seed: int=None,
                 data_format: Optional[str]=None, save_to_dir: Optional[str]=None,
                 save_prefix: str='', save_format: str='png', follow_links: bool=False):

        if data_format is None:
            data_format = K.image_data_format()
        self.directory = Path(directory)
        self.image_data_generator = image_data_generator
        self.target_size = target_size
        if color_mode not in {'rgb', 'grayscale'}:
            raise ValueError(f'Invalid color mode: {color_mode}; expected "rgb" or "grayscale".')
        self.color_mode = color_mode
        self.data_format = data_format
        n_channels = 3 if self.color_mode == 'rgb' else 1
        if self.data_format == 'channels_last':
            self.image_shape = (target_size['height'], target_size['width'], n_channels)
            self.mask_shape = (target_size['height'], target_size['width'], classes)
        else:
            self.image_shape = (n_channels, target_size['height'], target_size['width'])
            self.mask_shape = (classes, target_size['height'], target_size['width'])
        self.classes = classes
        self.save_to_dir = Path(save_to_dir) if save_to_dir is not None else None
        self.save_prefix = save_prefix
        self.save_format = save_format

        white_list_formats = {'.png', '.jpg', '.jpeg', '.bmp'}

        self.image_directory = Path(directory) / image_subfolder
        self.mask_directory = Path(directory) / mask_subfolder

        # first, count the number of sample_size and classes
        self.sample_size, self.image_filenames, self.mask_filenames = _list_valid_filenames_in_directory_segmentation(
            directory=directory,
            image_subfolder=image_subfolder,
            mask_subfolder= mask_subfolder,
            white_list_formats=white_list_formats
        )

        print(f'Found {self.sample_size} images')
        image_data_generator.sample_size = self.sample_size

        super().__init__(self.sample_size, batch_size, shuffle, seed)


    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros((len(index_array),) + self.image_shape, dtype=K.floatx())
        batch_y = np.zeros((len(index_array),) + self.mask_shape, dtype=K.floatx())
        grayscale = self.color_mode == 'grayscale'
        # build batch of image data
        for i, j in enumerate(index_array):
            fname = self.image_filenames[j]
            img = kpi.load_img(str(self.image_directory / fname),
                           grayscale=grayscale,
                           target_size=(self.target_size['height'], self.target_size['width']))
            x = kpi.img_to_array(img, data_format=self.data_format)
            # x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x

            fname = self.mask_filenames[j]
            img = load_mask(self.mask_directory / fname, target_size=self.target_size)
            y = kpi.img_to_array(img, data_format=self.data_format)
            y = self.image_data_generator.convert_to_onehot(y, self.classes, self.data_format)
            batch_y[i] = y

        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir is not None and self.save_to_dir.exists():
            for i, j in range(index_array):
                img = kpi.array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=j,
                                                                  hash=np.random.randint(1e4),
                                                                  format=self.save_format)
                img.save(self.save_to_dir/ fname)

        return batch_x, batch_y

    def next(self):
        """For python 2.x.

        # Returns
            The next batch.
        """
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)

