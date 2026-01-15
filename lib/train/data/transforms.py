import random
import numpy as np
import math
import cv2 as cv
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tvisf


class Transform:
    """A set of transformations, used for e.g. data augmentation.
    Args of constructor:
        transforms: An arbitrary number of transformations, derived from the TransformBase class.
                    They are applied in the order they are given.

    The Transform object can jointly transform images, bounding boxes and segmentation masks.
    This is done by calling the object with the following key-word arguments (all are optional).

    The following arguments are inputs to be transformed. They are either supplied as a single instance, or a list of instances.
        image  -  Image
        coords  -  2xN dimensional Tensor of 2D image coordinates [y, x]
        bbox  -  Bounding box on the form [x, y, w, h]
        mask  -  Segmentation mask with discrete classes

    The following parameters can be supplied with calling the transform object:
        joint [Bool]  -  If True then transform all images/coords/bbox/mask in the list jointly using the same transformation.
                         Otherwise each tuple (images, coords, bbox, mask) will be transformed independently using
                         different random rolls. Default: True.
        new_roll [Bool]  -  If False, then no new random roll is performed, and the saved result from the previous roll
                            is used instead. Default: True.

    Check the DiMPProcessing class for examples.
    """

    def __init__(self, *transforms):
        if len(transforms) == 1 and isinstance(transforms[0], (list, tuple)):
            transforms = transforms[0]
        self.transforms = transforms
        self._valid_inputs = ['image', 'coords', 'bbox', 'mask', 'att']
        self._valid_args = ['joint', 'new_roll']
        self._valid_all = self._valid_inputs + self._valid_args

    def __call__(self, **inputs):
        var_names = [k for k in inputs.keys() if k in self._valid_inputs]
        for v in inputs.keys():
            if v not in self._valid_all:
                raise ValueError('Incorrect input \"{}\" to transform. Only supports inputs {} and arguments {}.'.format(v, self._valid_inputs, self._valid_args))

        joint_mode = inputs.get('joint', True)
        new_roll = inputs.get('new_roll', True)

        if not joint_mode:
            out = zip(*[self(**inp) for inp in self._split_inputs(inputs)])
            return tuple(list(o) for o in out)

        out = {k: v for k, v in inputs.items() if k in self._valid_inputs}

        for t in self.transforms:
            out = t(**out, joint=joint_mode, new_roll=new_roll)
        if len(var_names) == 1:
            return out[var_names[0]]
        # Make sure order is correct
        return tuple(out[v] for v in var_names)

    def _split_inputs(self, inputs):
        var_names = [k for k in inputs.keys() if k in self._valid_inputs]
        split_inputs = [{k: v for k, v in zip(var_names, vals)} for vals in zip(*[inputs[vn] for vn in var_names])]
        for arg_name, arg_val in filter(lambda it: it[0]!='joint' and it[0] in self._valid_args, inputs.items()):
            if isinstance(arg_val, list):
                for inp, av in zip(split_inputs, arg_val):
                    inp[arg_name] = av
            else:
                for inp in split_inputs:
                    inp[arg_name] = arg_val
        return split_inputs

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class TransformBase:
    """Base class for transformation objects. See the Transform class for details."""
    def __init__(self):
        """2020.12.24 Add 'att' to valid inputs"""
        self._valid_inputs = ['image', 'coords', 'bbox', 'mask', 'att']
        self._valid_args = ['new_roll']
        self._valid_all = self._valid_inputs + self._valid_args
        self._rand_params = None

    def __call__(self, **inputs):
        # Split input
        input_vars = {k: v for k, v in inputs.items() if k in self._valid_inputs}
        input_args = {k: v for k, v in inputs.items() if k in self._valid_args}

        # Roll random parameters for the transform
        if input_args.get('new_roll', True):
            rand_params = self.roll()
            if rand_params is None:
                rand_params = ()
            elif not isinstance(rand_params, tuple):
                rand_params = (rand_params,)
            self._rand_params = rand_params

        outputs = dict()
        for var_name, var in input_vars.items():
            if var is not None:
                # if var_name == 'image':
                #     print(f"Transforming {var_name} with params {self._rand_params}")
                transform_func = getattr(self, 'transform_' + var_name)
                if var_name in ['coords', 'bbox']:
                    params = (self._get_image_size(input_vars),) + self._rand_params
                else:
                    params = self._rand_params
                if isinstance(var, (list, tuple)):
                    outputs[var_name] = [transform_func(x, *params) for x in var]
                else:
                    outputs[var_name] = transform_func(var, *params)
        return outputs

    def _get_image_size(self, inputs):
        im = None
        for var_name in ['image', 'mask']:
            if inputs.get(var_name) is not None:
                im = inputs[var_name]
                break
        if im is None:
            return None
        if isinstance(im, (list, tuple)):
            im = im[0]
        if isinstance(im, np.ndarray):
            return im.shape[:2]
        if torch.is_tensor(im):
            return (im.shape[-2], im.shape[-1])
        raise Exception('Unknown image type')

    def roll(self):
        return None

    def transform_image(self, image, *rand_params):
        """Must be deterministic"""
        return image

    def transform_coords(self, coords, image_shape, *rand_params):
        """Must be deterministic"""
        return coords

    def transform_bbox(self, bbox, image_shape, *rand_params):
        """Assumes [x, y, w, h]"""
        # Check if not overloaded
        if self.transform_coords.__code__ == TransformBase.transform_coords.__code__:
            return bbox

        coord = bbox.clone().view(-1,2).t().flip(0)

        x1 = coord[1, 0]
        x2 = coord[1, 0] + coord[1, 1]

        y1 = coord[0, 0]
        y2 = coord[0, 0] + coord[0, 1]

        coord_all = torch.tensor([[y1, y1, y2, y2], [x1, x2, x2, x1]])

        coord_transf = self.transform_coords(coord_all, image_shape, *rand_params).flip(0)
        tl = torch.min(coord_transf, dim=1)[0]
        sz = torch.max(coord_transf, dim=1)[0] - tl
        bbox_out = torch.cat((tl, sz), dim=-1).reshape(bbox.shape)
        return bbox_out

    def transform_mask(self, mask, *rand_params):
        """Must be deterministic"""
        return mask

    def transform_att(self, att, *rand_params):
        """2020.12.24 Added to deal with attention masks"""
        return att


class ToTensor(TransformBase):
    """Convert to a Tensor"""

    def transform_image(self, image):
        # handle numpy array
        if image.ndim == 2:
            image = image[:, :, None]

        image = torch.from_numpy(image.transpose((2, 0, 1)))
        # backward compatibility
        if isinstance(image, torch.ByteTensor):
            return image.float().div(255)
        else:
            return image

    def transfrom_mask(self, mask):
        if isinstance(mask, np.ndarray):
            return torch.from_numpy(mask)

    def transform_att(self, att):
        if isinstance(att, np.ndarray):
            return torch.from_numpy(att).to(torch.bool)
        elif isinstance(att, torch.Tensor):
            return att.to(torch.bool)
        else:
            raise ValueError ("dtype must be np.ndarray or torch.Tensor")


class ToTensorAndJitter(TransformBase):
    """Convert to a Tensor and jitter brightness"""
    def __init__(self, brightness_jitter=0.0, normalize=True):
        super().__init__()
        self.brightness_jitter = brightness_jitter
        self.normalize = normalize

    def roll(self):
        return np.random.uniform(max(0, 1 - self.brightness_jitter), 1 + self.brightness_jitter)

    def transform_image(self, image, brightness_factor):
        # handle numpy array
        image = torch.from_numpy(image.transpose((2, 0, 1)))

        # backward compatibility
        if self.normalize:
            return image.float().mul(brightness_factor/255.0).clamp(0.0, 1.0)
        else:
            return image.float().mul(brightness_factor).clamp(0.0, 255.0)

    def transform_mask(self, mask, brightness_factor):
        if isinstance(mask, np.ndarray):
            return torch.from_numpy(mask)
        else:
            return mask
    def transform_att(self, att, brightness_factor):
        if isinstance(att, np.ndarray):
            return torch.from_numpy(att).to(torch.bool)
        elif isinstance(att, torch.Tensor):
            return att.to(torch.bool)
        else:
            raise ValueError ("dtype must be np.ndarray or torch.Tensor")


class Normalize(TransformBase):
    """Normalize image"""
    def __init__(self, mean, std, inplace=False):
        super().__init__()
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def transform_image(self, image):
        """支持任意通道数的归一化"""
        dims = list(image.size())
        channels = dims[0]
        
        # 检查是否有对应通道数的统计量
        if channels == len(self.mean) and channels == len(self.std):
            # 如果统计量的通道数匹配，直接使用
            return tvisf.normalize(image, self.mean, self.std, self.inplace)
        else:
            # 如果不匹配，需要处理
            
            # 方法A：扩展统计量
            if channels > len(self.mean):
                # 假设self.mean只有3个值，但图像有8个通道
                # 重复或扩展统计量
                repeat_times = (channels + len(self.mean) - 1) // len(self.mean)
                extended_mean = (self.mean * repeat_times)[:channels]
                extended_std = (self.std * repeat_times)[:channels]
                return tvisf.normalize(image, extended_mean, extended_std, self.inplace)
            
            # 方法B：使用通用归一化函数
            return self._normalize_any_channels(image)

    def _normalize_any_channels(self, image, mean=None, std=None, inplace=False):
        """支持任意通道数的归一化实现"""
        if mean is None:
            mean = self.mean
        if std is None:
            std = self.std
        
        channels = image.shape[0]
        
        # 确保mean和std的长度与通道数一致
        if len(mean) != channels:
            if len(mean) == 3 and channels > 3:
                # 假设是RGB+其他通道的情况
                # 对RGB使用前3个值，对其他通道使用平均值
                rgb_mean = mean[:3]
                other_mean_value = sum(mean) / len(mean)  # 使用平均值
                extended_mean = list(rgb_mean) + [other_mean_value] * (channels - 3)
                mean = extended_mean
            
            elif len(mean) == 1:
                # 如果只有一个值，重复到所有通道
                mean = [mean[0]] * channels
            else:
                # 其他情况，使用默认值
                mean = [0.5] * channels
        
        if len(std) != channels:
            if len(std) == 3 and channels > 3:
                rgb_std = std[:3]
                other_std_value = sum(std) / len(std)
                extended_std = list(rgb_std) + [other_std_value] * (channels - 3)
                std = extended_std
            elif len(std) == 1:
                std = [std[0]] * channels
            else:
                std = [0.5] * channels
        
        # 手动实现归一化
        if not inplace:
            image = image.clone()
        
        # 对每个通道分别归一化
        for c in range(channels):
            image[c] = (image[c] - mean[c]) / std[c]
        
        return image


class ToGrayscale(TransformBase):
    """Converts image to grayscale with probability"""
    def __init__(self, probability = 0.5):
        super().__init__()
        self.probability = probability
        self.color_weights = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)

    def roll(self):
        return random.random() < self.probability

    def transform_image(self, image, do_grayscale):
        if do_grayscale:
            if torch.is_tensor(image):
                raise NotImplementedError('Implement torch variant.')
            if image.shape[-1] == 6:
                img_rgb = image[:,:,:3]
                img_dte = image[:, :, 3:]
                img_gray = cv.cvtColor(img_rgb, cv.COLOR_RGB2GRAY)
                if (img_rgb == img_dte).min() == True:
                    return np.stack([img_gray, img_gray, img_gray, img_gray, img_gray, img_gray], axis=2)
                else:
                    return np.stack([img_gray, img_gray, img_gray, img_dte[:,:,0], img_dte[:,:,1], img_dte[:,:,2]], axis=2)
            elif image.shape[-1] == 3:
                img_gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
                return np.stack([img_gray, img_gray, img_gray], axis=2)
            elif image.shape[-1] == 8:
                # 对于8通道图像，只将前3个RGB通道转换为灰度，保持其他通道不变
                img_rgb = image[:,:,:3]
                img_other = image[:, :, 3:]
                img_gray = cv.cvtColor(img_rgb, cv.COLOR_RGB2GRAY)
                gray_stacked = np.stack([img_gray, img_gray, img_gray], axis=2)
                return np.concatenate([gray_stacked, img_other], axis=2)
            elif image.shape[-1] == 16:
                # 禁用灰度
                return image
            else:
                raise ValueError("Error of image shape")
        return image


class ToBGR(TransformBase):
    """Converts image to BGR"""
    def transform_image(self, image):
        if torch.is_tensor(image):
            raise NotImplementedError('Implement torch variant.')
        img_bgr = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        return img_bgr


class RandomHorizontalFlip(TransformBase):
    """Horizontally flip image randomly with a probability p."""
    def __init__(self, probability = 0.5):
        super().__init__()
        self.probability = probability

    def roll(self):
        return random.random() < self.probability

    def transform_image(self, image, do_flip):
        if do_flip:
            if torch.is_tensor(image):
                return image.flip((2,))
            return np.fliplr(image).copy()
        return image

    def transform_coords(self, coords, image_shape, do_flip):
        if do_flip:
            coords_flip = coords.clone()
            coords_flip[1,:] = (image_shape[1] - 1) - coords[1,:]
            return coords_flip
        return coords

    def transform_mask(self, mask, do_flip):
        if do_flip:
            if torch.is_tensor(mask):
                return mask.flip((-1,))
            return np.fliplr(mask).copy()
        return mask

    def transform_att(self, att, do_flip):
        if do_flip:
            if torch.is_tensor(att):
                return att.flip((-1,))
            return np.fliplr(att).copy()
        return att


class RandomHorizontalFlip_Norm(RandomHorizontalFlip):
    """Horizontally flip image randomly with a probability p.
    The difference is that the coord is normalized to [0,1]"""
    def __init__(self, probability = 0.5):
        super().__init__()
        self.probability = probability

    def transform_coords(self, coords, image_shape, do_flip):
        """we should use 1 rather than image_shape"""
        if do_flip:
            coords_flip = coords.clone()
            coords_flip[1,:] = 1 - coords[1,:]
            return coords_flip
        return coords
