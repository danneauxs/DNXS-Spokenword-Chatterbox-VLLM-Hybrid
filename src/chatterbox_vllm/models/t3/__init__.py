from .t3 import T3VllmModel, SPEECH_TOKEN_OFFSET
from .t3_turbo import T3TurboVllmModel
from .t3_turbo import SPEECH_TOKEN_OFFSET as TURBO_SPEECH_TOKEN_OFFSET
from vllm import ModelRegistry
from vllm.transformers_utils.tokenizer_base import TokenizerRegistry

ModelRegistry.register_model("ChatterboxT3", T3VllmModel)
ModelRegistry.register_model("ChatterboxT3Turbo", T3TurboVllmModel)
TokenizerRegistry.register("EnTokenizer", "chatterbox_vllm.models.t3.entokenizer", "EnTokenizer")
TokenizerRegistry.register("MtlTokenizer", "chatterbox_vllm.models.t3.mtltokenizer", "MTLTokenizer")
TokenizerRegistry.register("TurboTokenizer", "chatterbox_vllm.models.t3.turbotokenizer", "TurboTokenizer")
