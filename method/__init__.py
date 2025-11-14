
from .resa import ReSA
from .AdaDim import AdaDim
from .AdaDim_base import AdaDim_Base

def get_method(args):
    if args.loss_type == "Resa":
        return ReSA(args)
    elif args.loss_type == "AdaDim_base":
        return AdaDim_Base(args)
    else:
        return AdaDim(args)




 