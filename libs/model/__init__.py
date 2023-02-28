from .abpn import *
from .rlfn import *
from .vapsr import *
from .mobilesr import *
from .hpinet import *
from .common import rmbn
from .abpn2 import *

NETWORKS = ['RLFN', 'ABPN', 'VAPSR', 'MOBILESR', 'HPINET', 'ABPNV2']

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
    elif name == "ABPNV2":
        return ABPNv3()
    else:
        raise NotImplementedError(name)