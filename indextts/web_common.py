"""Small presentation helpers, without feature-service imports."""
import functools
import gradio as gr

def ui_errors(fn):
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            raise gr.Error(str(exc)) from exc
    return wrapped


def choices(library):
    return [(f"{item['displayName']} · {item['voiceId']}", item["voiceId"]) for item in library.items()]
