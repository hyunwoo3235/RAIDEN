from typing import TYPE_CHECKING

from transformers.file_utils import _LazyModule, is_torch_available

_import_structure = {
    "configuration_vits": ["VITSConfig"],
    "tokenization_vits": ["VITSTokenizer"],
}

if is_torch_available():
    _import_structure["modeling_vits"] = [
        "VITSModel",
        "VITSPreTrainedModel",
        "VITSForWaveformGeneration",
        "VITSMultiPeriodDiscriminator",
    ]


if TYPE_CHECKING:
    from .configuration_vits import VITSConfig
    from .tokenization_vits import VITSTokenizer

    if is_torch_available():
        from .modeling_vits import (
            VITSModel,
            VITSPreTrainedModel,
            VITSForWaveformGeneration,
            VITSMultiPeriodDiscriminator,
        )

else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__, globals()["__file__"], _import_structure, module_spec=__spec__
    )
