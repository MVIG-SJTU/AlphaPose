# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by Jiefeng Li (jeff.lee.sjtu@gmail.com)
# -----------------------------------------------------

import os
import torch
import torch.distributed as dist


def init_dist(opt):
    """Initialize distributed computing environment."""
    opt.ngpus_per_node = torch.cuda.device_count()

    torch.cuda.set_device(opt.gpu)

    if opt.launcher == 'pytorch':
        _init_dist_pytorch(opt)
    elif opt.launcher == 'mpi':
        _init_dist_mpi(opt)
    elif opt.launcher == 'slurm':
        _init_dist_slurm(opt)
    else:
        raise ValueError('Invalid launcher type: {}'.format(opt.launcher))


def _init_dist_pytorch(opt, **kwargs):
    """Set up environment."""
    # TODO: use local_rank instead of rank % num_gpus
    opt.rank = opt.rank * opt.ngpus_per_node + opt.gpu
    opt.world_size = opt.world_size
    dist.init_process_group(backend=opt.dist_backend, init_method=opt.dist_url,
                            world_size=opt.world_size, rank=opt.rank)
    print(f"{opt.dist_url}, ws:{opt.world_size}, rank:{opt.rank}")

    if opt.rank % opt.ngpus_per_node == 0:
        opt.log = True
    else:
        opt.log = False


def _init_dist_slurm(opt, port=23348, **kwargs):
    """Set up slurm environment."""
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id % num_gpus)
    if '[' in node_list:
        beg = node_list.find('[')
        pos1 = node_list.find('-', beg)
        if pos1 < 0:
            pos1 = 1000
        pos2 = node_list.find(',', beg)
        if pos2 < 0:
            pos2 = 1000
        node_list = node_list[:min(pos1, pos2)].replace('[', '')
    addr = node_list[8:].replace('-', '.')
    os.environ['MASTER_PORT'] = str(port)
    os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['RANK'] = str(proc_id)

    opt.ngpus_per_node = num_gpus
    opt.rank = int(proc_id)
    opt.rank = proc_id * num_gpus + opt.gpu
    opt.world_size = int(ntasks) * num_gpus

    print(f"tcp://{node_list}:{port}, ws:{opt.world_size}, rank:{opt.rank}, proc_id:{proc_id}")
    dist.init_process_group(backend=opt.dist_backend,
                            init_method=f'tcp://{node_list}:{port}',
                            world_size=opt.world_size,
                            rank=opt.rank)
    if opt.rank == 0:
        opt.log = True
    else:
        opt.log = False


def _init_dist_mpi(backend, **kwargs):
    raise NotImplementedError
