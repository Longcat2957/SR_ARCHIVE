from .abpn import *
from .rlfn import *
from .vapsr import *
from .mobilesr import *
from .hpinet import *
from .common import rmbn

NETWORKS = ['RLFN', 'ABPN', 'VAPSR', 'MOBILESR', 'HPINET']

def get_network(name:str):
    if name == 'RLFN':
        return RLFN()
    elif name == "ABPN":
        return ABPN()
    elif name == "VAPSR":
        return VAPSR()
    elif name == "MOBILESR":
        return MOBILESR()
    elif name == "HPINET":
        return HPINET()
    else:
        raise NotImplementedError(name)