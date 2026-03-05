from __future__ import annotations

import os
from pathlib import Path

import pytest

from adaparse.parsers.nougat import NougatParser
from adaparse.parsers.nougat import NougatParserConfig
from tests.utils.helpers import get_nougat_checkpoint

pytestmark = [pytest.mark.fs, pytest.mark.integration]


def _get_test_pdfs(pdf_dir_raw: str | None, max_pdfs_raw: str | None) -> list[Path]:
    """Resolve PDF test inputs from CLI/env-configured values."""
    if not pdf_dir_raw:
        pytest.skip(
            'Provide --nq-pdf-dir or set ADAPARSE_NQ_TEST_PDF_DIR to a PDF directory.'
        )

    pdf_dir = Path(pdf_dir_raw).expanduser()
    if not pdf_dir.exists() or not pdf_dir.is_dir():
        pytest.skip(f'ADAPARSE_NQ_TEST_PDF_DIR is not a directory: {pdf_dir}')

    pdf_files = sorted(pdf_dir.glob('*.pdf'))
    if not pdf_files:
        pytest.skip(f'No PDFs found in directory: {pdf_dir}')

    max_pdfs_raw = max_pdfs_raw or os.getenv('ADAPARSE_NQ_TEST_MAX_PDFS', '3')
    try:
        max_pdfs = max(1, int(max_pdfs_raw))
    except ValueError:
        max_pdfs = 3

    return pdf_files[:max_pdfs]


def test_nougat_parser_smoke_on_local_pdf_directory(pytestconfig) -> None:
    """Smoke test: verify NougatParser runs in the local environment."""
    checkpoint = get_nougat_checkpoint()
    if not checkpoint or not checkpoint.exists():
        pytest.skip('Nougat checkpoint not found. Set NOUGAT_CHECKPOINT in .adaparse.env.')

    pdf_dir_raw = pytestconfig.getoption('--nq-pdf-dir') or os.getenv(
        'ADAPARSE_NQ_TEST_PDF_DIR'
    )
    max_pdfs_raw = pytestconfig.getoption('--nq-max-pdfs') or os.getenv(
        'ADAPARSE_NQ_TEST_MAX_PDFS'
    )
    log_dir_raw = pytestconfig.getoption('--nq-log-dir') or os.getenv(
        'ADAPARSE_NQ_TEST_LOG_DIR', './tests/logs'
    )

    pdf_files = _get_test_pdfs(pdf_dir_raw, max_pdfs_raw)
    logs_path = Path(log_dir_raw).expanduser()

    try:
        parser = NougatParser(
            NougatParserConfig(
                checkpoint=checkpoint,
                batchsize=1,
                num_workers=0,
                nougat_logs_path=logs_path,
            )
        )
    except Exception as exc:  # pragma: no cover - env-dependent import/runtime failures
        pytest.skip(f'NougatParser initialization failed in this environment: {exc}')

    docs = parser.parse([str(path) for path in pdf_files])
    assert docs is not None, 'Parser returned None (likely runtime failure).'
    assert len(docs) > 0, 'Parser returned an empty document list.'
    assert any((doc.get('text') or '').strip() for doc in docs), (
        'Parser ran but produced no non-empty text output.'
    )
