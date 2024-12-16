from models.coil import COIL
from models.der import DER
from models.ewc import EWC

from models.gem import GEM
from models.icarl import iCaRL
from models.lwf import LwF
from models.replay import Replay
from models.bic import BiC
from models.podnet import PODNet
from models.wa import WA
#from models.finetune import Finetune
from models.ssl import SSL
from models.ssl_finetune import SSL_Finetune
from models.ssl_lwf import SSL_LWF
from models.cassle import Simsiam
from models.replay_unlabel import Replay_unlabel
from models.unlabel import Unlabel
def get_model(model_name, args):
    name = model_name.lower()
    if name == 'icarl':
        return iCaRL(args)
    elif name == 'bic':
        return BiC(args)
    elif name == 'podnet':
        return PODNet(args)
    elif name == "lwf":
        return LwF(args)
    elif name == "ewc":
        return EWC(args)
    elif name == "wa":
        return WA(args)
    elif name == "der":
        return DER(args)

    elif name == "replay":
        return Replay(args)
    elif name == "gem":
        return GEM(args)
    elif name == "coil":
        return COIL(args)
    elif name == "ssl":
        return SSL(args)
    elif name == "ssl_finetune":
        return SSL_Finetune(args)
    elif name == "ssl_lwf":
        return SSL_LWF(args)
    elif name == "simsam":
        return Simsiam(args)
    elif name == "replay_unlabel":
        return Replay_unlabel(args)
    elif name == "unlabel":
        return Unlabel(args)

    else:
        assert 0
