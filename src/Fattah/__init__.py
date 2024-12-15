import sys
from transformers.utils import _LazyModule

# Define the structure of your imports relative to the current directory
_import_structure = {
    "configuration_fattah": ["fattahConfig"],
    "modeling_fattah": [
        "fattahForCausalLM",
        "fattahForSequenceClassification",
        "fattahModel",
        "fattahPreTrainedModel",
    ],
}

# Set up lazy loading of the modules
sys.modules[__name__] = _LazyModule(
    __name__, globals()["__file__"], _import_structure, module_spec=__spec__
)
