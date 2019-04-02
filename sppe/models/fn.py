from mxnet import ndarray


def _try_load_parameters(self, filename=None, model=None, ctx=None, allow_missing=False,
                         ignore_extra=False):
        def getblock(parent, name):
            if len(name) == 1:
                if name[0].isnumeric():
                    return parent[int(name[0])]
                else:
                    return getattr(parent, name[0])
            else:
                if name[0].isnumeric():
                    return getblock(parent[int(name[0])], name[1:])
                else:
                    return getblock(getattr(parent, name[0]), name[1:])
        if filename is not None:
            loaded = ndarray.load(filename)
        else:
            loaded = model._collect_params_with_prefix()
        params = self._collect_params_with_prefix()
        if not loaded and not params:
            return

        if not any('.' in i for i in loaded.keys()):
            # legacy loading
            del loaded
            self.collect_params().load(
                filename, ctx, allow_missing, ignore_extra, self.prefix)
            return
        '''
        for name in params.keys():
            if name not in loaded:
                name_split = name.split('.')
                block = getblock(self, name_split)
                block.initialize(ctx=ctx)
        '''
        for name in loaded:
            if name in params:
                if params[name].shape != loaded[name].shape:
                    continue
                params[name]._load_init(loaded[name], ctx)


def _load_from_pytorch(self, filename, ctx=None):
    import torch
    from mxnet import nd
    loaded = torch.load(filename)
    params = self._collect_params_with_prefix()

    new_params = {}

    for name in loaded:
        if 'bn' in name or 'batchnorm' in name or '.downsample.1.' in name:
            if 'weight' in name:
                mxnet_name = name.replace('weight', 'gamma')
            elif 'bias' in name:
                mxnet_name = name.replace('bias', 'beta')
            else:
                mxnet_name = name
            new_params[mxnet_name] = nd.array(loaded[name].cpu().data.numpy())
        else:
            new_params[name] = nd.array(loaded[name].cpu().data.numpy())

    for name in new_params:
        if name not in params:
            print(f'==={name}===')
            raise Exception
        if name in params:
            params[name]._load_init(new_params[name], ctx=ctx)
