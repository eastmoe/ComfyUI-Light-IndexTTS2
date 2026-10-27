from torch import nn

from indextts.s2mel.modules.diffusion_transformer import DiT


class CFM(nn.Module):
    def __init__(self, args):
        super().__init__()
        if args.dit_type != "DiT":
            raise ValueError(f"Unsupported diffusion type: {args.dit_type}")
        self.estimator = DiT(args)
