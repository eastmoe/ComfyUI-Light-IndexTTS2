import torch
from torch import nn

from indextts.utils.compat import torch_load


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    return t_act * s_act


def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


class MyModel(nn.Module):
    def __init__(self, args, use_emovec=False, use_gpt_latent=False):
        super().__init__()
        from indextts.s2mel.modules.flow_matching import CFM
        from indextts.s2mel.modules.length_regulator import InterpolateRegulator

        length_regulator = InterpolateRegulator(
            channels=args.length_regulator.channels,
            sampling_ratios=args.length_regulator.sampling_ratios,
            is_discrete=args.length_regulator.is_discrete,
            in_channels=args.length_regulator.in_channels if hasattr(args.length_regulator, "in_channels") else None,
            vector_quantize=args.length_regulator.vector_quantize if hasattr(args.length_regulator, "vector_quantize") else False,
            codebook_size=args.length_regulator.content_codebook_size,
            n_codebooks=args.length_regulator.n_codebooks if hasattr(args.length_regulator, "n_codebooks") else 1,
            quantizer_dropout=args.length_regulator.quantizer_dropout if hasattr(args.length_regulator, "quantizer_dropout") else 0.0,
            f0_condition=args.length_regulator.f0_condition if hasattr(args.length_regulator, "f0_condition") else False,
            n_f0_bins=args.length_regulator.n_f0_bins if hasattr(args.length_regulator, "n_f0_bins") else 512,
        )

        modules = {
            "cfm": CFM(args),
            "length_regulator": length_regulator,
        }
        if use_gpt_latent:
            modules["gpt_layer"] = nn.Sequential(
                nn.Linear(1280, 256),
                nn.Linear(256, 128),
                nn.Linear(128, 1024),
            )
        self.models = nn.ModuleDict(modules)

    def forward_gpt(self, x):
        return self.models["gpt_layer"](x)


def load_checkpoint2(
    model,
    optimizer,
    path,
    load_only_params=True,
    ignore_modules=None,
    is_distributed=False,
    load_ema=False,
):
    ignore_modules = ignore_modules or []
    state = torch_load(path, map_location="cpu")
    params = state["net"]
    if load_ema and "ema" in state:
        print("Loading EMA")
        for key in model.models:
            for i, param_name in enumerate(params[key]):
                if "input_pos" in param_name:
                    continue
                params[key][param_name] = state["ema"][key][0][i].clone()

    for key in model.models:
        if key not in params or key in ignore_modules:
            continue
        if not is_distributed:
            for name in list(params[key].keys()):
                if name.startswith("module."):
                    params[key][name[len("module.") :]] = params[key][name]
                    del params[key][name]

        model_state_dict = model.models[key].state_dict()
        filtered_state_dict = {
            name: value
            for name, value in params[key].items()
            if name in model_state_dict and value.shape == model_state_dict[name].shape
        }
        skipped_keys = set(params[key].keys()) - set(filtered_state_dict.keys())
        if skipped_keys:
            print(f"Warning: skipped loading shape-mismatched keys: {skipped_keys}")
        print(f"{key} loaded")
        model.models[key].load_state_dict(filtered_state_dict, strict=False)

    model.eval()
    return model, optimizer, 0, 0
