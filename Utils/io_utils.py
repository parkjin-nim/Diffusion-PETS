import os
import sys
import yaml
import json
import torch
import random
import warnings
import importlib
import numpy as np


def load_yaml_config(path):
    with open(path) as f:
        config = yaml.full_load(f)
    return config

def save_config_to_yaml(config, path):
    assert path.endswith('.yaml')
    with open(path, 'w') as f:
        f.write(yaml.dump(config))
        f.close()

def save_dict_to_json(d, path, indent=None):
    json.dump(d, open(path, 'w'), indent=indent)

def load_dict_from_json(path):
    return json.load(open(path, 'r'))

def write_args(args, path):
    args_dict = dict((name, getattr(args, name)) for name in dir(args)if not name.startswith('_'))
    with open(path, 'a') as args_file:
        args_file.write('==> torch version: {}\n'.format(torch.__version__))
        args_file.write('==> cudnn version: {}\n'.format(torch.backends.cudnn.version()))
        args_file.write('==> Cmd:\n')
        args_file.write(str(sys.argv))
        args_file.write('\n==> args:\n')
        for k, v in sorted(args_dict.items()):
            args_file.write('  %s: %s\n' % (str(k), str(v)))
        args_file.close()

def seed_everything(seed, cudnn_deterministic=False):
    """
    Function that sets seed for pseudo-random number generators in:
    pytorch, numpy, python.random
    
    Args:
        seed: the integer value seed for global random state
    """
    if seed is not None:
        print(f"Global seed set to {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False

    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

def merge_opts_to_config(config, opts):
    def modify_dict(c, nl, v):
        if len(nl) == 1:
            c[nl[0]] = type(c[nl[0]])(v)
        else:
            # print(nl)
            c[nl[0]] = modify_dict(c[nl[0]], nl[1:], v)
        return c

    if opts is not None and len(opts) > 0:
        assert len(opts) % 2 == 0, "each opts should be given by the name and values! The length shall be even number!"
        for i in range(len(opts) // 2):
            name = opts[2*i]
            value = opts[2*i+1]
            config = modify_dict(config, name.split('.'), value)
    return config 

def modify_config_for_debug(config):
    config['dataloader']['num_workers'] = 0
    config['dataloader']['batch_size'] = 1
    return config

def get_model_parameters_info(model):
    # for mn, m in model.named_modules():
    parameters = {'overall': {'trainable': 0, 'non_trainable': 0, 'total': 0}}
    for child_name, child_module in model.named_children():
        parameters[child_name] = {'trainable': 0, 'non_trainable': 0}
        for pn, p in child_module.named_parameters():
            if p.requires_grad:
                parameters[child_name]['trainable'] += p.numel()
            else:
                parameters[child_name]['non_trainable'] += p.numel()
        parameters[child_name]['total'] = parameters[child_name]['trainable'] + parameters[child_name]['non_trainable']
        
        parameters['overall']['trainable'] += parameters[child_name]['trainable']
        parameters['overall']['non_trainable'] += parameters[child_name]['non_trainable']
        parameters['overall']['total'] += parameters[child_name]['total']
    
    # format the numbers
    def format_number(num):
        K = 2**10
        M = 2**20
        G = 2**30
        if num > G: # K
            uint = 'G'
            num = round(float(num)/G, 2)
        elif num > M:
            uint = 'M'
            num = round(float(num)/M, 2)
        elif num > K:
            uint = 'K'
            num = round(float(num)/K, 2)
        else:
            uint = ''
        
        return '{}{}'.format(num, uint)
    
    def format_dict(d):
        for k, v in d.items():
            if isinstance(v, dict):
                format_dict(v)
            else:
                d[k] = format_number(v)
    
    format_dict(parameters)
    return parameters

def format_seconds(seconds):
    h = int(seconds // 3600)
    m = int(seconds // 60 - h * 60)
    s = int(seconds % 60)

    d = int(h // 24)
    h = h - d * 24

    if d == 0:
        if h == 0:
            if m == 0:
                ft = '{:02d}s'.format(s)
            else:
                ft = '{:02d}m:{:02d}s'.format(m, s)
        else:
           ft = '{:02d}h:{:02d}m:{:02d}s'.format(h, m, s)
 
    else:
        ft = '{:d}d:{:02d}h:{:02d}m:{:02d}s'.format(d, h, m, s)

    return ft

def instantiate_from_config(config):
    if config is None:
        return None
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    # config["target"]는 클래스의 전체 경로를 나타내는 문자열입니다. 예를 들어, "models.MyModel"과 같이 되어 있을 수 있습니다. 이 문자열을 모듈과 클래스 이름으로 분리하여 해당 클래스를 가져옵니다.
    # rsplit(".", 1)은 문자열을 오른쪽에서부터 "."을 기준으로 최대 1번 분리하여 모듈과 클래스 이름을 얻습니다. 
    # 예를 들어, "Utils.Data_utils.real_datasets.CustomDataset"은 "Utils.Data_utils.real_datasets"와 "CustomDataset"로 분리됩니다.
    module, cls = config["target"].rsplit(".", 1)
    # importlib.import_module(module, package=None)은 모듈을 동적으로 가져오는 함수입니다. module은 가져올 모듈의 이름을 나타내는 문자열입니다. 예를 들어, "Utils.Data_utils.real_datasets"와 같이 되어 있을 수 있습니다.
    # getattr(importlib.import_module(module, package=None), cls)는 가져온 모듈에서 cls라는 이름의 클래스를 가져오는 함수입니다. 예를 들어, "CustomDataset"이라는 이름의 클래스를 가져옵니다.
    cls = getattr(importlib.import_module(module, package=None), cls)
    # config.get("params", dict())는 config 딕셔너리에서 "params" 키에 해당하는 값을 가져옵니다. 만약 "params" 키가 존재하지 않으면 빈 딕셔너리를 반환합니다. 이 값은 클래스의 생성자에 전달될 매개변수들을 포함하는 딕셔너리입니다.
    # cls(**config.get("params", dict()))는 가져온 클래스를 인스턴스화하는 코드입니다. 클래스의 생성자에 config 딕셔너리에서 "params" 키에 해당하는 값을 매개변수로 전달하여 객체를 생성합니다. 예를 들어, CustomDataset 클래스의 생성자에 필요한 매개변수들을 config 딕셔너리의 "params" 키에 포함시켜서 객체를 생성할 수 있습니다.
    return cls(**config.get("params", dict()))

def class_from_string(class_name):
    module, cls = class_name.rsplit(".", 1)
    cls = getattr(importlib.import_module(module, package=None), cls)
    return cls

def get_all_file(dir, end_with='.h5'):
    if isinstance(end_with, str):
        end_with = [end_with]
    filenames = []
    for root, dirs, files in os.walk(dir):
        for f in files:
            for ew in end_with:
                if f.endswith(ew):
                    filenames.append(os.path.join(root, f))
                    break
    return filenames

def get_sub_dirs(dir, abs=True):
    sub_dirs = os.listdir(dir)
    if abs:
        sub_dirs = [os.path.join(dir, s) for s in sub_dirs]
    return sub_dirs

def get_model_buffer(model):
    state_dict = model.state_dict()
    buffers_ = {}
    params_ = {n: p for n, p in model.named_parameters()}

    for k in state_dict:
        if k not in params_:
            buffers_[k] = state_dict[k]
    return buffers_
