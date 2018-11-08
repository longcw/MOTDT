from .mot_seq import MOTSeq


__factory = {
    # 'kitti': KITTISeq,
    'mot': MOTSeq,
}


def get_names():
    return tuple(__factory.keys())


def init_dataset(name, *args, **kwargs):
    if name not in get_names():
        raise KeyError("Unknown dataset: {}".format(name))
    return __factory[name](*args, **kwargs)
