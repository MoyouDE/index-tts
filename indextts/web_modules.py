"""Dependency-free selection of validation features."""
MODULES = ("producer", "emotion", "audition")


def parse_modules(value=None):
    names = list(MODULES) if value is None else (value.split(",") if isinstance(value, str) else list(value))
    names = [name.strip() if isinstance(name, str) else name for name in names]
    if not names or any(not isinstance(name, str) or name not in MODULES for name in names):
        raise ValueError("modules 需要包含 producer、emotion、audition 中的一个或多个名称")
    if len(names) != len(set(names)):
        raise ValueError("modules 不允许重复名称")
    return tuple(name for name in MODULES if name in names)
