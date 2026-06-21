from .configuration_xvla import XVLAConfig

__all__ = [
    "XVLAConfig",
    "XVLAPolicy",
    "XVLAAddDomainIdProcessorStep",
    "XVLAImageNetNormalizeProcessorStep",
    "XVLAImageToFloatProcessorStep",
]


def __getattr__(name: str):
    if name == "XVLAPolicy":
        from .modeling_xvla import XVLAPolicy

        return XVLAPolicy
    if name == "XVLAAddDomainIdProcessorStep":
        from .processor_xvla import XVLAAddDomainIdProcessorStep

        return XVLAAddDomainIdProcessorStep
    if name == "XVLAImageNetNormalizeProcessorStep":
        from .processor_xvla import XVLAImageNetNormalizeProcessorStep

        return XVLAImageNetNormalizeProcessorStep
    if name == "XVLAImageToFloatProcessorStep":
        from .processor_xvla import XVLAImageToFloatProcessorStep

        return XVLAImageToFloatProcessorStep
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
