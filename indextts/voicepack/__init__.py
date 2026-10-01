"""IndexTTS voice-pack creation and safe loading APIs."""

__all__ = ["VoicePack", "VoicePackBuilder", "VoicePackError", "load_voicepack"]


def __getattr__(name):
    if name == "VoicePackBuilder":
        from .builder import VoicePackBuilder
        return VoicePackBuilder
    if name in {"VoicePack", "VoicePackError", "load_voicepack"}:
        from . import archive
        return getattr(archive, name)
    raise AttributeError(name)
