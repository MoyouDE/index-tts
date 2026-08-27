"""Reader-focused IndexTTS runtime with no reference-audio encoders."""

from .emotion import EmotionProvider, ExplicitEmotionProvider, QwenEmotionProvider

__all__ = ["EmotionProvider", "ExplicitEmotionProvider", "QwenEmotionProvider", "ReaderRuntime"]


def __getattr__(name):
    if name == "ReaderRuntime":
        from .engine import ReaderRuntime

        return ReaderRuntime
    raise AttributeError(name)
