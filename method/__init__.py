
from .resa import ReSA
from .AdaDim import AdaDim

def get_method(args):
    if args.loss_type == "Resa":
        return ReSA(args)
    else:
        return AdaDim(args)




 