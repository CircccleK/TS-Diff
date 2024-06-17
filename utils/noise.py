import random
from numbers import Number
from os.path import join

import numpy as np
import torch
import yaml
from torch import nn

from utils import get_root_logger


class RawPacker:
    def __init__(self, cfa='bayer'):
        self.cfa = cfa

    def pack_raw_bayer(self, cfa_img):
        # pack Bayer image to 4 channels
        img_shape = cfa_img.shape
        H = img_shape[0]
        W = img_shape[1]

        out = np.stack((cfa_img[0:H:2, 0:W:2],  # RGBG
                        cfa_img[0:H:2, 1:W:2],
                        cfa_img[1:H:2, 1:W:2],
                        cfa_img[1:H:2, 0:W:2]), axis=0).astype(np.float32)
        return out

    def pack_raw_xtrans(self, cfa_img):
        # pack X-Trans image to 9 channels
        img_shape = cfa_img.shape
        H = (img_shape[0] // 6) * 6
        W = (img_shape[1] // 6) * 6

        out = np.zeros((9, H // 3, W // 3), dtype=np.float32)

        # 0 R
        out[0, 0::2, 0::2] = cfa_img[0:H:6, 0:W:6]
        out[0, 0::2, 1::2] = cfa_img[0:H:6, 4:W:6]
        out[0, 1::2, 0::2] = cfa_img[3:H:6, 1:W:6]
        out[0, 1::2, 1::2] = cfa_img[3:H:6, 3:W:6]

        # 1 G
        out[1, 0::2, 0::2] = cfa_img[0:H:6, 2:W:6]
        out[1, 0::2, 1::2] = cfa_img[0:H:6, 5:W:6]
        out[1, 1::2, 0::2] = cfa_img[3:H:6, 2:W:6]
        out[1, 1::2, 1::2] = cfa_img[3:H:6, 5:W:6]

        # 1 B
        out[2, 0::2, 0::2] = cfa_img[0:H:6, 1:W:6]
        out[2, 0::2, 1::2] = cfa_img[0:H:6, 3:W:6]
        out[2, 1::2, 0::2] = cfa_img[3:H:6, 0:W:6]
        out[2, 1::2, 1::2] = cfa_img[3:H:6, 4:W:6]

        # 4 R
        out[3, 0::2, 0::2] = cfa_img[1:H:6, 2:W:6]
        out[3, 0::2, 1::2] = cfa_img[2:H:6, 5:W:6]
        out[3, 1::2, 0::2] = cfa_img[5:H:6, 2:W:6]
        out[3, 1::2, 1::2] = cfa_img[4:H:6, 5:W:6]

        # 5 B
        out[4, 0::2, 0::2] = cfa_img[2:H:6, 2:W:6]
        out[4, 0::2, 1::2] = cfa_img[1:H:6, 5:W:6]
        out[4, 1::2, 0::2] = cfa_img[4:H:6, 2:W:6]
        out[4, 1::2, 1::2] = cfa_img[5:H:6, 5:W:6]

        out[5, :, :] = cfa_img[1:H:3, 0:W:3]
        out[6, :, :] = cfa_img[1:H:3, 1:W:3]
        out[7, :, :] = cfa_img[2:H:3, 0:W:3]
        out[8, :, :] = cfa_img[2:H:3, 1:W:3]
        return out

    def unpack_raw_bayer(self, img):
        # unpack 4 channels to Bayer image
        img4c = img
        _, h, w = img.shape

        H = int(h * 2)
        W = int(w * 2)

        cfa_img = np.zeros((H, W), dtype=np.float32)

        cfa_img[0:H:2, 0:W:2] = img4c[0, :, :]
        cfa_img[0:H:2, 1:W:2] = img4c[1, :, :]
        cfa_img[1:H:2, 1:W:2] = img4c[2, :, :]
        cfa_img[1:H:2, 0:W:2] = img4c[3, :, :]

        return cfa_img

    def unpack_raw_xtrans(self, img):
        img9c = img
        _, h, w = img.shape

        H = int(h * 3)
        W = int(w * 3)

        cfa_img = np.zeros((H, W), dtype=np.float32)

        # 0 R
        cfa_img[0:H:6, 0:W:6] = img9c[0, 0::2, 0::2]
        cfa_img[0:H:6, 4:W:6] = img9c[0, 0::2, 1::2]
        cfa_img[3:H:6, 1:W:6] = img9c[0, 1::2, 0::2]
        cfa_img[3:H:6, 3:W:6] = img9c[0, 1::2, 1::2]

        # 1 G
        cfa_img[0:H:6, 2:W:6] = img9c[1, 0::2, 0::2]
        cfa_img[0:H:6, 5:W:6] = img9c[1, 0::2, 1::2]
        cfa_img[3:H:6, 2:W:6] = img9c[1, 1::2, 0::2]
        cfa_img[3:H:6, 5:W:6] = img9c[1, 1::2, 1::2]

        # 1 B
        cfa_img[0:H:6, 1:W:6] = img9c[2, 0::2, 0::2]
        cfa_img[0:H:6, 3:W:6] = img9c[2, 0::2, 1::2]
        cfa_img[3:H:6, 0:W:6] = img9c[2, 1::2, 0::2]
        cfa_img[3:H:6, 4:W:6] = img9c[2, 1::2, 1::2]

        # 4 R
        cfa_img[1:H:6, 2:W:6] = img9c[3, 0::2, 0::2]
        cfa_img[2:H:6, 5:W:6] = img9c[3, 0::2, 1::2]
        cfa_img[5:H:6, 2:W:6] = img9c[3, 1::2, 0::2]
        cfa_img[4:H:6, 5:W:6] = img9c[3, 1::2, 1::2]

        # 5 B
        cfa_img[2:H:6, 2:W:6] = img9c[4, 0::2, 0::2]
        cfa_img[1:H:6, 5:W:6] = img9c[4, 0::2, 1::2]
        cfa_img[4:H:6, 2:W:6] = img9c[4, 1::2, 0::2]
        cfa_img[5:H:6, 5:W:6] = img9c[4, 1::2, 1::2]

        cfa_img[1:H:3, 0:W:3] = img9c[5, :, :]
        cfa_img[1:H:3, 1:W:3] = img9c[6, :, :]
        cfa_img[2:H:3, 0:W:3] = img9c[7, :, :]
        cfa_img[2:H:3, 1:W:3] = img9c[8, :, :]

        return cfa_img

    def pack_raw(self, cfa_img):
        if self.cfa == 'bayer':
            out = self.pack_raw_bayer(cfa_img)
        elif self.cfa == 'xtrans':
            out = self.pack_raw_xtrans(cfa_img)
        else:
            raise NotImplementedError
        return out

    def unpack_raw(self, img):
        if self.cfa == 'bayer':
            out = self.unpack_raw_bayer(img)
        elif self.cfa == 'xtrans':
            out = self.unpack_raw_xtrans(img)
        else:
            raise NotImplementedError
        return out


  # LED
def _pack_bayer(raw):
    h, w = raw.size(-2), raw.size(-1)
    out = torch.cat((raw[..., None, 0:h:2, 0:w:2],  # R
                     raw[..., None, 0:h:2, 1:w:2],  # G1
                     raw[..., None, 1:h:2, 1:w:2],  # B
                     raw[..., None, 1:h:2, 0:w:2]  # G2
                     ), dim=-3)
    return out


def _normal_batch(scale=1.0, loc=0.0, shape=(1,)):
    return torch.randn(shape) * scale + loc


def _uniform_batch(min_, max_, shape=(1,)):
    return torch.rand(shape) * (max_ - min_) + min_


def _randint_batch(min_, max_, shape=(1,)):
    return torch.randint(min_, max_, shape)


def shot_noise(x, k):
    return torch.poisson(x / k) * k - x


def gaussian_noise(x, scale, loc=0):
    return torch.randn_like(x) * scale + loc


def tukey_lambda_noise(x, scale, t_lambda=1.4):
    def tukey_lambda_ppf(p, t_lambda):
        assert not torch.any(t_lambda == 0.0)
        return 1 / t_lambda * (p ** t_lambda - (1 - p) ** t_lambda)

    epsilon = 1e-10

    U = torch.rand_like(x) * (1 - 2 * epsilon) + epsilon
    Y = tukey_lambda_ppf(U, t_lambda) * scale

    return Y


def quant_noise(x, q):
    return (torch.rand_like(x) - 0.5) * q


def row_noise(x, scale, loc=0):
    if x.dim() == 4:
        B, _, H, W = x.shape
        # noise = (torch.randn((B, H * 2, 1)) * scale + loc).repeat((1, 1, W * 2))
        # return _pack_bayer(noise)
        noise = (torch.randn((B, H, 1)) * scale + loc).repeat((1, 1, W))
        return noise.unsqueeze(1)
    elif x.dim() == 5:
        B, T, _, H, W = x.shape
        noise = (torch.randn((B, T, H * 2, 1)) * scale + loc).repeat((1, 1, 1, W * 2))
        return _pack_bayer(noise)
    else:
        raise NotImplementedError()


class NoiseModelBase:  # base class
    def __init__(self):
        self.k_min = None
        self.k_max = None
        self.row_sigmas = None
        self.row_biases = None
        self.row_slopes = None
        self.tukey_lambdas = None
        self.read_sigmas = None
        self.read_biases = None
        self.read_slopes = None

        self.noise_type = None
        self.log_K = None
        self.current_camera_params = None
        self.current_camera_idx = None

    def __call__(self, img, vcam_id=None, params=None): # Sony的参数
        if not torch.is_tensor(img):
            img = torch.tensor(img)
            if img.dim() == 3:
                img = img.unsqueeze(0)

        self.cur_batch_size = img.size(0)

        if vcam_id is not None:
            self.current_camera_idx = vcam_id * torch.ones((self.cur_batch_size,), dtype=torch.long) \

        ratio = torch.tensor(np.random.uniform(low=100, high=300))


        white_level = 16383
        black_level = 800
        scale = torch.tensor(white_level - black_level)

        img = torch.tensor(img) * scale.unsqueeze(0) / ratio.unsqueeze(0)

        K = self.sample_overall_system_gain()
        noise = {}
        noise_params = {'isp_dgain': ratio, 'scale': scale}

        img = img.clone().detach()

        # shot noise
        if 'p' in self.noise_type:
            _shot_noise = shot_noise(img, K)
            noise['shot'] = _shot_noise
            noise_params['shot'] = K.squeeze()
        # read noise
        if 'g' in self.noise_type:
            read_param = self.sample_read_sigma_Gaussian()
            _read_noise = gaussian_noise(img, read_param)
            noise['read'] = _read_noise
            noise_params['read'] = read_param.squeeze()
        elif 't' in self.noise_type:
            tukey_lambda = self.sample_tukey_lambda()
            read_param = self.sample_read_sigma_TukeyLambda()
            _read_noise = tukey_lambda_noise(img, read_param, tukey_lambda)
            noise['read'] = _read_noise
            noise_params['read'] = {
                'sigma': read_param,
                'tukey_lambda': tukey_lambda
            }
        # row noise
        if 'r' in self.noise_type:
            row_param = self.sample_row_sigma()
            _row_noise = row_noise(img, row_param)
            noise['row'] = _row_noise
            noise_params['row'] = row_param.squeeze()
        # quant noise
        if 'q' in self.noise_type:
            _quant_noise = quant_noise(img, 1)
            noise['quant'] = _quant_noise
        # # color bias
        # if 'c' in self.noise_type:
        #     color_bias = self.sample_color_bias()
        #     noise['color_bias'] = color_bias

        img_lq = self.add_noise(img, noise, noise_params)

        return img_lq.squeeze()

    def sample_overall_system_gain(self):
        if self.current_camera_idx is not None:
            log_K_max = torch.log(self.k_max)
            log_K_min = torch.log(self.k_min)
        else:
            log_K_max = torch.log(self.current_camera_params['Kmax'])
            log_K_min = torch.log(self.current_camera_params['Kmin'])
        log_K = _uniform_batch(log_K_min, log_K_max, (self.cur_batch_size, 1, 1, 1))
        self.log_K = log_K
        return torch.exp(log_K)

    def sample_read_sigma_Gaussian(self):
        if self.current_camera_idx is not None:
            slope = self.read_slopes[self.current_camera_idx]
            bias = self.read_biases[self.current_camera_idx]
            sigma = self.read_sigmas[self.current_camera_idx]
        else:
            slope = self.current_camera_params['Gaussian']['slope']
            bias = self.current_camera_params['Gaussian']['bias']
            sigma = self.current_camera_params['Gaussian']['sigma']
        mu = self.log_K.squeeze() * slope + bias
        sample = _normal_batch(sigma, mu, (self.cur_batch_size, ))
        return torch.exp(sample).reshape(self.log_K.shape)

    def sample_read_sigma_TukeyLambda(self):
        if self.current_camera_idx is not None:
            slope = self.read_slopes[self.current_camera_idx]
            bias = self.read_biases[self.current_camera_idx]
            sigma = self.read_sigmas[self.current_camera_idx]
        else:
            slope = self.current_camera_params['TukeyLambda']['slope']
            bias = self.current_camera_params['TukeyLambda']['bias']
            sigma = self.current_camera_params['TukeyLambda']['sigma']
        mu = self.log_K.squeeze() * slope + bias
        sample = _normal_batch(sigma, mu, (self.cur_batch_size, ))
        return torch.exp(sample).reshape(self.log_K.shape)

    def sample_tukey_lambda(self):
        if self.current_camera_idx is not None:
            tukey_lambdas = self.tukey_lambdas[self.current_camera_idx].reshape(self.cur_batch_size, 1, 1, 1)
        else:
            index = _randint_batch(0, len(self.current_camera_params['TukeyLambda']['lam']), shape=(self.cur_batch_size, ))
            tukey_lambdas = self.current_camera_params['TukeyLambda']['lam'][index].reshape(self.cur_batch_size, 1, 1, 1)
        return tukey_lambdas

    def sample_row_sigma(self):
        if self.current_camera_idx is not None:
            slope = self.row_slopes[self.current_camera_idx]
            bias = self.row_biases[self.current_camera_idx]
            sigma = self.row_sigmas[self.current_camera_idx]
        else:
            slope = self.current_camera_params['Row']['slope']
            bias = self.current_camera_params['Row']['bias']
            sigma = self.current_camera_params['Row']['sigma']
        mu = self.log_K.squeeze() * slope + bias
        sample = _normal_batch(sigma, mu, (self.cur_batch_size, ))
        return torch.exp(sample).reshape(self.log_K.squeeze(-3).shape)

    # def sample_color_bias(self):
    #     count = len(self.current_camera_params['ColorBias'])
    #     i_range = (self.current_camera_params['Kmax'] - self.current_camera_params['Kmin']) / count
    #     index = ((torch.exp(self.log_K.squeeze()) - self.current_camera_params['Kmin']) // i_range).long()
    #     color_bias = self.current_camera_params['ColorBias'][index]
    #     color_bias = color_bias.reshape(4, 1, 1, 1)
    #     return color_bias

    @staticmethod
    def add_noise(img, noise, noise_params):
        ratio = noise_params['isp_dgain']
        scale = noise_params['scale']
        for n in noise.values():
            img += n
        img /= scale
        img = img * ratio
        return torch.clamp(img, max=1.0)


# Only support baseline noise models: G / G+P / G+P*
def param_dict_to_tensor_dict(p_dict):
    def to_tensor_dict(p_dict):
        for k, v in p_dict.items():
            if isinstance(v, list) or isinstance(v, Number):
                p_dict[k] = nn.Parameter(torch.tensor(v), False)
            elif isinstance(v, dict):
                p_dict[k] = to_tensor_dict(v)
        return p_dict
    return to_tensor_dict(p_dict)


class NoiseModel(NoiseModelBase):
    def __init__(self, cameras=None, cfa='bayer'):
        super().__init__()
        assert cfa in ['bayer', 'xtrans']
        self.cameras = cameras

        self.param_dir = join('camera_params')

        self.camera_params = {}

        with open(join(self.param_dir, cameras + '_params.yaml')) as f:
            self.camera_params = yaml.safe_load(f)

        self.noise_type = self.camera_params['noise_type']

        if self.cameras == 'Virtual':
            self.sample_virtual_cameras()
        else:
            self.current_camera_params = param_dict_to_tensor_dict(self.camera_params['camera_params'])

        logger = get_root_logger()
        logger.info(f"NoiseModel with {format(self.param_dir)}")
        logger.info(f"cameras {format(self.cameras)}")
        logger.info(f"using noise model {format(self.noise_type)}")

        self.raw_packer = RawPacker(cfa)

    def sample_virtual_cameras(self):
        self.virtual_camera_count = self.camera_params['virtual_camera_count']
        self.sample_strategy = self.camera_params['sample_strategy']
        self.shuffle = self.camera_params.get('shuffle', False)
        self.param_ranges = self.camera_params['param_ranges']

        # sampling strategy
        sample = self.split_range if self.sample_strategy == 'coverage' else self.uniform_range

        print('Current Using Cameras: ', [f'IC{i}' for i in range(self.virtual_camera_count)])

        self.k_max = torch.tensor(self.param_ranges['K'][0])
        self.k_min = torch.tensor(self.param_ranges['K'][1])

        # read noise
        if 'g' in self.noise_type:
            read_slope_range = self.param_ranges['Gaussian']['slope']
            read_bias_range = self.param_ranges['Gaussian']['bias']
            read_sigma_range = self.param_ranges['Gaussian']['sigma']
        elif 't' in self.noise_type:
            read_slope_range = self.param_ranges['TukeyLambda']['slope']
            read_bias_range = self.param_ranges['TukeyLambda']['bias']
            read_sigma_range = self.param_ranges['TukeyLambda']['sigma']
            read_lambda_range = self.param_ranges['TukeyLambda']['lambda']
            self.tukey_lambdas = sample(self.virtual_camera_count, read_lambda_range, self.shuffle)
            self.tukey_lambdas = nn.Parameter(self.tukey_lambdas, False)
        if 'g' in self.noise_type or 't' in self.noise_type:
            self.read_slopes = sample(self.virtual_camera_count, read_slope_range, self.shuffle)
            self.read_biases = sample(self.virtual_camera_count, read_bias_range, self.shuffle)
            self.read_sigmas = sample(self.virtual_camera_count, read_sigma_range, self.shuffle)
            self.read_slopes = nn.Parameter(self.read_slopes, False)
            self.read_biases = nn.Parameter(self.read_biases, False)
            self.read_sigmas = nn.Parameter(self.read_sigmas, False)

        # row noise
        if 'r' in self.noise_type:
            row_slope_range = self.param_ranges['Row']['slope']
            row_bias_range = self.param_ranges['Row']['bias']
            row_sigma_range = self.param_ranges['Row']['sigma']
            self.row_slopes = sample(self.virtual_camera_count, row_slope_range, self.shuffle)
            self.row_biases = sample(self.virtual_camera_count, row_bias_range, self.shuffle)
            self.row_sigmas = sample(self.virtual_camera_count, row_sigma_range, self.shuffle)
            self.row_slopes = nn.Parameter(self.row_slopes, False)
            self.row_biases = nn.Parameter(self.row_biases, False)
            self.row_sigmas = nn.Parameter(self.row_sigmas, False)

        # # color bias
        # if 'c' in self.noise_type:
        #     self.color_bias_count = self.param_ranges['ColorBias']['count']
        #     ## ascend sigma
        #     color_bias_sigmas = self.split_range_overlap(self.color_bias_count,
        #                                                  self.param_ranges['ColorBias']['sigma'],
        #                                                  overlap=0.1)
        #     self.color_biases = torch.tensor(np.array([
        #         [
        #             random.uniform(*self.param_ranges['ColorBias']['bias']) + \
        #             torch.randn(4).numpy() * random.uniform(*color_bias_sigmas[i]).cpu().numpy()
        #             for _ in range(self.color_bias_count)
        #         ] for i in range(self.virtual_camera_count)
        #     ]), device=self.device)
        #     self.color_biases = nn.Parameter(self.color_biases, False)

    @staticmethod
    def split_range(splits, range_, shuffle=True):
        length = range_[1] - range_[0]
        i_length = length / (splits - 1)
        results = [range_[0] + i_length * i for i in range(splits)]
        if shuffle:
            random.shuffle(results)
        return torch.tensor(results)

    @staticmethod
    def uniform_range(splits, range_, shuffle=True):
        results = [random.uniform(*range_) for _ in range(splits)]
        if shuffle:
            random.shuffle(results)
        return torch.tensor(results)

    # @staticmethod
    # def split_range_overlap(splits, range_, overlap=0.5, device='cuda'):
    #     length = range_[1] - range_[0]
    #     i_length = length / (splits * (1 - overlap) + overlap)
    #     results = []
    #     for i in range(splits):
    #         start = i_length * (1 - overlap) * i
    #         results.append([start, start + i_length])
    #     return torch.tensor(results, device=device)