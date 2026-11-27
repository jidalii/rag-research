from .base import Method
from .hipporag import HippoRAG
from .raptor import Raptor
from .simple import FullContext, LLMOnly, LLMOnlyDecomposed, Oracle
from .vanilla_dos_slot import DOSRAG, SlotRAG, VanillaRAG

# name -> class. Teammates: register GraphRAG / vanilla RAG here.
REGISTRY: dict[str, type[Method]] = {
    LLMOnly.name: LLMOnly,
    VanillaRAG.name: VanillaRAG,
    DOSRAG.name: DOSRAG,
    SlotRAG.name: SlotRAG,
    FullContext.name: FullContext,
    Raptor.name: Raptor,
    HippoRAG.name: HippoRAG,
    Oracle.name: Oracle,
    LLMOnlyDecomposed.name: LLMOnlyDecomposed,
}


def get_method(name: str) -> type[Method]:
    if name not in REGISTRY:
        raise ValueError(f"Unknown method {name!r}; available: {sorted(REGISTRY)}")
    return REGISTRY[name]
