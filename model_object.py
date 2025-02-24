from torch.nn import CrossEntropyLoss
from torch.optim import *
from Net.MultiLayerFusion import *
from Net.api import *
from loss_function import joint_loss, LatentFusionLoss
from utils import get_scheduler
models = {
'IHFNet':{
        'Name': 'IHFNet',
        'Model': IHFNet,
        'Loss': LatentFusionLoss,
        'Optimizer': AdamW,
        'batch_size': 8,
        'Lr': 0.0001,
        'Epoch': 200,
        'w1': 0.2,
        'w2': 0.01,
        'Run': run_main_2,
        'Scheduler': get_scheduler,
    },
'IHFNetWithoutCMIM':{
        'Name': 'IHFNet',
        'Model': IHFNet_Without_HMCAM,
        'Loss': LatentFusionLoss,
        'Optimizer': AdamW,
        'batch_size': 8,
        'Lr': 0.0001,
        'Epoch': 200,
        'w1': 0.2,
        'w2': 0.01,
        'Run': run_main_2,
        'Scheduler': get_scheduler,
}

}
