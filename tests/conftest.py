from __future__ import annotations


def pytest_addoption(parser) -> None:
    """Register custom CLI options for local Nougat smoke testing."""
    parser.addoption(
        '--nq-pdf-dir',
        action='store',
        default=None,
        help='Directory containing PDFs for Nougat smoke test.',
    )
    parser.addoption(
        '--nq-max-pdfs',
        action='store',
        default=None,
        help='Maximum number of PDFs to parse (default: 3).',
    )
    parser.addoption(
        '--nq-log-dir',
        action='store',
        default=None,
        help='Directory for Nougat logs (default: ./tests/logs).',
    )
