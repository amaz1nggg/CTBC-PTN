from typing import Optional
import os.path as osp
import json
import yaml
import nni


class SimpleParam:
    # 这个类实现了一个简单的参数获取功能。开发者可以通过配置不同的source参数（'nni'、'local:FILE_NAME'、'default'）
    # 来获取对应的超参集合，并且还支持对读取到的参数进行预处理。

    def __init__(self, local_dir: str = 'param', default: Optional[dict] = None):
        # 初始化函数，设置默认的本地目录和默认参数集合
        if default is None:
            default = dict()

        self.local_dir = local_dir
        self.default = default

    def __call__(self, source: str, preprocess: str = 'none'):
        # 调用函数，根据source参数获取对应的超参集合，并进行预处理

        if source == 'nni':
            # 如果source为'nni'，则使用nni.get_next_parameter()获取下一个超参集合，并与默认参数集合合并
            return {**self.default, **nni.get_next_parameter()}

        if source.startswith('local'):
            # 如果source以'local'开头，则解析出文件名，并根据文件后缀选择合适的解析函数进行解析
            ts = source.split(':')
            assert len(ts) == 2, 'local parameter file should be specified in a form of `local:FILE_NAME`'
            path = ts[-1]
            path = osp.join(self.local_dir, path)

            if path.endswith('.json'):
                loaded = parse_json(path)
            elif path.endswith('.yaml') or path.endswith('.yml'):
                loaded = parse_yaml(path)
            else:
                raise Exception('Invalid file name. Should end with .yaml or .json.')

            if preprocess == 'nni':
                # 如果preprocess为'nni'，则对读取到的参数集合进行预处理
                loaded = preprocess_nni(loaded)

            # 将读取到的参数集合与默认参数集合合并
            return {**self.default, **loaded}

        if source == 'default':
            # 如果source为'default'，则返回默认参数集合
            return self.default

        # 如果source不是'nni'、'local:FILE_NAME'、'default'中的任何一个，则抛出异常
        raise Exception('invalid source')


def preprocess_nni(params: dict):
    # 这段代码实现了一个简单的函数，用于对NNI训练任务中读取的参数进行处理。
    # 在参数名称以<prefix>/<param_name>的形式给出时，该函数会将其转换为只包含参数名称的形式。
    # 这个函数可以帮助开发者更方便地使用NNI训练超参，并且也是NNI中常用的参数处理方式之一。

    def process_key(key: str):
        # 内部函数，用于处理参数名称
        xs = key.split('/')
        if len(xs) == 3:
            return xs[1]
        elif len(xs) == 1:
            return key
        else:
            raise Exception('Unexpected param name ' + key)

    # 使用字典推导式，将params中的每个参数名称经过process_key函数处理后，与原参数值一起构建新的字典
    return {
        process_key(k): v for k, v in params.items()
    }

def parse_yaml(path: str):
# 这段代码实现了一个简单的函数，用于解析指定路径的YAML文件，并返回解析后的内容。
# 该函数首先读取文件的内容，然后使用yaml.load()方法将内容解析成Python对象。
# 这个函数可以帮助开发者在程序中方便地读取和处理YAML文件的内容。
    content = open(path).read()
    return yaml.load(content, Loader=yaml.Loader)


def parse_json(path: str):
# 这段代码实现了一个简单的函数，用于解析指定路径的JSON文件，并返回解析后的内容。
# 该函数首先读取文件的内容，然后使用json.loads()方法将内容解析成Python对象。
# 这个函数可以帮助开发者在程序中方便地读取和处理JSON文件的内容。
    content = open(path).read()
    return json.loads(content)
