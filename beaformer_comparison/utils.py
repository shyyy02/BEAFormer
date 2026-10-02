from torch import nn

def load_model_state_dict(model: nn.Module, state_dict):
    if isinstance(model, nn.DataParallel):
        model = model.module

    if hasattr(model, "net"):
        model = model.net

    ret = model.load_state_dict(state_dict, strict=False)
    return ret

def get_model_state_dict(model):
    if isinstance(model, nn.DataParallel):
        model = model.module

    if hasattr(model, "net"):
        model = model.net

    return model.state_dict()