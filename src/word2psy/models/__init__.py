"""Lazy model registry — imports only happen when a model class is accessed."""


def __getattr__(name):
    _registry = {
        "LexicalNormsModel": ".lexical_norms",
        "WordformModel": ".wordform",
        "FastTextModel": ".fasttext_embed",
        "Word2VecModel": ".word2vec",
        "GPT2SurprisalModel": ".gpt2_surprisal",
        "SentimentModel": ".sentiment",
        "EmotionModel": ".emotion",
        "ReadabilityModel": ".readability",
        "SentenceEmbedModel": ".sentence_embed",
        "CLIPTextModel": ".clip_text",
        "CLAPTextModel": ".clap_text",
        "EBindTextModel": ".ebind_text",
    }
    if name in _registry:
        import importlib

        module = importlib.import_module(_registry[name], __package__)
        return getattr(module, name)
    if name == "__all__":
        return list(_registry.keys())
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
