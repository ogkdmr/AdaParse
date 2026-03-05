"""The parsers module storing different PDF parsers."""

from __future__ import annotations

from importlib import import_module
from typing import Any
from typing import TYPE_CHECKING

from adaparse.registry import registry

if TYPE_CHECKING:
    from adaparse.parsers.adaparse import AdaParse
    from adaparse.parsers.adaparse import AdaParseConfig
    from adaparse.parsers.marker import MarkerParser
    from adaparse.parsers.marker import MarkerParserConfig
    from adaparse.parsers.nougat import NougatParser
    from adaparse.parsers.nougat import NougatParserConfig
    from adaparse.parsers.pymupdf import PyMuPDFParser
    from adaparse.parsers.pymupdf import PyMuPDFParserConfig
    from adaparse.parsers.pypdf import PyPDFParser
    from adaparse.parsers.pypdf import PyPDFParserConfig
    from adaparse.parsers.tesseract import TesseractParser
    from adaparse.parsers.tesseract import TesseractParserConfig

    ParserConfigTypes = (
        AdaParseConfig
        | MarkerParserConfig
        | NougatParserConfig
        | PyMuPDFParserConfig
        | PyPDFParserConfig
        | TesseractParserConfig
    )
    ParserTypes = (
        AdaParse
        | MarkerParser
        | NougatParser
        | PyMuPDFParser
        | PyPDFParser
        | TesseractParser
    )
else:
    ParserConfigTypes = Any
    ParserTypes = Any

_ParserTypes = tuple[type[Any], type[Any]]

_STRATEGY_SPECS: dict[str, tuple[str, str, str]] = {
    'adaparse': ('adaparse.parsers.adaparse', 'AdaParseConfig', 'AdaParse'),
    'marker': ('adaparse.parsers.marker', 'MarkerParserConfig', 'MarkerParser'),
    'nougat': ('adaparse.parsers.nougat', 'NougatParserConfig', 'NougatParser'),
    'pymupdf': ('adaparse.parsers.pymupdf', 'PyMuPDFParserConfig', 'PyMuPDFParser'),
    'pypdf': ('adaparse.parsers.pypdf', 'PyPDFParserConfig', 'PyPDFParser'),
    'tesseract': (
        'adaparse.parsers.tesseract',
        'TesseractParserConfig',
        'TesseractParser',
    ),
}

_EXPORT_SPECS: dict[str, tuple[str, str]] = {
    'AdaParse': ('adaparse.parsers.adaparse', 'AdaParse'),
    'AdaParseConfig': ('adaparse.parsers.adaparse', 'AdaParseConfig'),
    'MarkerParser': ('adaparse.parsers.marker', 'MarkerParser'),
    'MarkerParserConfig': ('adaparse.parsers.marker', 'MarkerParserConfig'),
    'NougatParser': ('adaparse.parsers.nougat', 'NougatParser'),
    'NougatParserConfig': ('adaparse.parsers.nougat', 'NougatParserConfig'),
    'PyMuPDFParser': ('adaparse.parsers.pymupdf', 'PyMuPDFParser'),
    'PyMuPDFParserConfig': ('adaparse.parsers.pymupdf', 'PyMuPDFParserConfig'),
    'PyPDFParser': ('adaparse.parsers.pypdf', 'PyPDFParser'),
    'PyPDFParserConfig': ('adaparse.parsers.pypdf', 'PyPDFParserConfig'),
    'TesseractParser': ('adaparse.parsers.tesseract', 'TesseractParser'),
    'TesseractParserConfig': ('adaparse.parsers.tesseract', 'TesseractParserConfig'),
}

__all__ = [
    'ParserConfigTypes',
    'ParserTypes',
    'get_parser',
    *_EXPORT_SPECS.keys(),
]


def _resolve_strategy(name: str) -> _ParserTypes | None:
    spec = _STRATEGY_SPECS.get(name)
    if spec is None:
        return None
    module_name, config_name, parser_name = spec
    module = import_module(module_name)
    config_cls = getattr(module, config_name)
    parser_cls = getattr(module, parser_name)
    return config_cls, parser_cls


def __getattr__(name: str) -> Any:
    spec = _EXPORT_SPECS.get(name)
    if spec is None:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    module_name, symbol_name = spec
    module = import_module(module_name)
    value = getattr(module, symbol_name)
    globals()[name] = value
    return value


# This is a workaround to support optional registration.
# Make a function to combine the config and instance initialization
# since the registry only accepts functions with hashable arguments.
def _factory_fn(**kwargs: dict[str, Any]) -> ParserTypes:
    name = kwargs.get('name', '')
    strategy = _resolve_strategy(name)  # type: ignore[arg-type]
    if not strategy:
        raise ValueError(
            f'Unknown parser name: {name}.'
            f' Available: {set(_STRATEGY_SPECS.keys())}',
        )

    # Get the config and classes
    config_cls, cls = strategy

    return cls(config_cls(**kwargs))


def get_parser(
    kwargs: dict[str, Any],
    register: bool = False,
) -> ParserTypes:
    """Get the instance based on the kwargs.

    Currently supports the following strategies:
    - adaparse
    - marker
    - oreo
    - nougat
    - pymupdf
    - pypdf
    - tesseract

    Parameters
    ----------
    kwargs : dict[str, Any]
        The configuration. Contains a `name` argument
        to specify the strategy to use.
    register : bool, optional
        Register the instance for warmstart. Caches the
        instance based on the kwargs, by default False.

    Returns
    -------
    ParserTypes
        The instance.

    Raises
    ------
    ValueError
        If the `name` is unknown.
    """
    # Create and register the instance
    if register:
        registry.register(_factory_fn)
        return registry.get(_factory_fn, **kwargs)

    return _factory_fn(**kwargs)
