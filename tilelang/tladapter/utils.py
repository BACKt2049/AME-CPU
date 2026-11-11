"""Common utilities for future MLIR-backed TileLang adapters."""


def _format_pass_option_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _build_pipeline_text(pass_name: str, anchor: str | None = None, **options) -> str:
    inner = pass_name
    if options:
        opts = ",".join(
            f"{key.replace('_', '-')}={_format_pass_option_value(value)}"
            for key, value in options.items()
        )
        inner = f"{pass_name}{{{opts}}}"
    if anchor:
        inner = f"{anchor}({inner})"
    return inner


def _get_native_module():
    try:
        import tilelang.tladapter as _tladapter
    except ImportError as err:
        raise ImportError(
            "TileLang MLIR adapters require a native module. "
            "Build TileLang with the riscv/MLIR integration enabled."
        ) from err
    native = getattr(_tladapter, "_native", None)
    if native is None:
        raise RuntimeError(
            "tilelang.tladapter native module not found; "
            "the MLIR integration scaffold is present but the native binding is not built yet."
        )
    return native


class Pipeline:
    def __init__(self):
        self._pp = _get_native_module().PassPipeline()

    def add(self, pass_or_text, **options):
        if isinstance(pass_or_text, str):
            if options:
                raise ValueError(
                    "options are not supported with raw pipeline text; "
                    "embed them in the string or use a pass descriptor"
                )
            self._pp.add(pass_or_text)
        elif isinstance(pass_or_text, _PassDescriptor):
            self._pp.add(pass_or_text._make_pipeline_text(**options))
        else:
            raise TypeError(f"expected str or pass descriptor, got {type(pass_or_text)}")
        return self

    def enable_ir_printing(self):
        self._pp.enable_ir_printing()
        return self

    def run(self, mlir_str: str) -> str:
        return self._pp.run(mlir_str)

    def __str__(self):
        return str(self._pp)

    def __repr__(self):
        return repr(self._pp)


class _PassDescriptor:
    def __init__(self, pass_name: str, anchor: str | None = None, **default_options):
        self._pass_name = pass_name
        self._anchor = anchor
        self._default_options = default_options

    def _make_pipeline_text(self, **extra_options):
        merged = {**self._default_options, **extra_options}
        return _build_pipeline_text(self._pass_name, self._anchor, **merged)

    def __call__(self, *args, **options):
        text = self._make_pipeline_text(**options)
        pp = _get_native_module().PassPipeline()
        pp.add(text)
        if not args:
            return pp
        value = args[0]
        result = pp.run(str(value) if not isinstance(value, str) else value)
        if isinstance(value, str):
            return result
        return result


def pass_fn(pass_name: str, anchor: str | None = None, **options):
    return _PassDescriptor(pass_name, anchor=anchor, **options)
