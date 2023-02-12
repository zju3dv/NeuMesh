def build_framework(args, framework):
    if framework == "NeuMesh":
        from .neumesh import get_model
    elif framework == "NeuS":
        from .neus import get_model
    else:
        raise NotImplementedError
    return get_model(args)
