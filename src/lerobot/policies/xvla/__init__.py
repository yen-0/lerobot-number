from .configuration_xvla import XVLAConfig
from .processor_xvla import (
    XVLAAddDomainIdProcessorStep,
    XVLAImageNetNormalizeProcessorStep,
    XVLAImageToFloatProcessorStep,
)

__all__ = [
    "XVLAConfig",
    "XVLAAddDomainIdProcessorStep",
    "XVLAImageNetNormalizeProcessorStep",
    "XVLAImageToFloatProcessorStep",
]


def __getattr__(name: str):
    if name == "XVLAPolicy":
        from .modeling_xvla import XVLAPolicy

        return XVLAPolicy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
