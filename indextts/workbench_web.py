"""Legacy imports for local-library and audition controls, loaded on demand."""
def __getattr__(name):
    if name in {"choices", "ui_errors"}:
        from . import web_common
        return getattr(web_common, name)
    if name == "library_controls":
        from .library_web import library_controls
        return library_controls
    if name == "audition_controls":
        from .audition_web import build_page
        return build_page
    raise AttributeError(name)
