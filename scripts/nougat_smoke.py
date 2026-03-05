from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Iterable

# Ensure local package import works when executed as:
#   python scripts/nougat_smoke.py ...
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adaparse.parsers.nougat import NougatParser
from adaparse.parsers.nougat import NougatParserConfig

ENV_BASENAME = '.adaparse.env'
_ASSIGN = re.compile(r'^\s*(?:export\s+)?([A-Za-z_]\w*)\s*=\s*(.*)\s*$')


def _find_env_file(start: Path) -> Path | None:
    p = start.resolve()
    while True:
        candidate = p / ENV_BASENAME
        if candidate.exists():
            return candidate
        if p.parent == p:
            return None
        p = p.parent


def _load_env_file(path: Path | None) -> None:
    if path is None or not path.exists():
        return

    for raw in path.read_text(encoding='utf-8').splitlines():
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        match = _ASSIGN.match(line)
        if not match:
            continue

        key, value = match.group(1), match.group(2)
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]
        elif ' #' in value:
            value = value.split(' #', 1)[0].rstrip()

        os.environ[key] = os.path.expanduser(os.path.expandvars(value))


def _collect_pdfs(pdf_paths: Iterable[str], pdf_dir: str | None, max_pdfs: int) -> list[Path]:
    selected: list[Path] = []

    for raw in pdf_paths:
        p = Path(raw).expanduser().resolve()
        if p.is_file() and p.suffix.lower() == '.pdf':
            selected.append(p)

    if pdf_dir:
        directory = Path(pdf_dir).expanduser().resolve()
        if not directory.exists() or not directory.is_dir():
            raise FileNotFoundError(f'--pdf-dir is not a directory: {directory}')
        selected.extend(sorted(directory.glob('*.pdf')))

    # Deduplicate while preserving order.
    deduped: list[Path] = []
    seen: set[str] = set()
    for p in selected:
        key = str(p)
        if key not in seen:
            seen.add(key)
            deduped.append(p)

    if not deduped:
        raise ValueError('No PDFs found. Provide --pdf and/or --pdf-dir containing .pdf files.')

    return deduped[: max(1, max_pdfs)]


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Standalone Nougat smoke runner (no pytest).'
    )
    parser.add_argument(
        '--pdf',
        action='append',
        default=[],
        help='Path to a PDF file. Can be passed multiple times.',
    )
    parser.add_argument(
        '--pdf-dir',
        type=str,
        default=None,
        help='Directory containing PDFs to parse.',
    )
    parser.add_argument(
        '--max-pdfs',
        type=int,
        default=3,
        help='Maximum number of PDFs to parse.',
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Nougat checkpoint path. Defaults to NOUGAT_CHECKPOINT env/.adaparse.env.',
    )
    parser.add_argument(
        '--log-dir',
        type=str,
        default='./tests/logs',
        help='Directory for Nougat logs.',
    )
    parser.add_argument(
        '--out-jsonl',
        type=str,
        default='./tests/logs/nougat_smoke_output.jsonl',
        help='Output JSONL path for parsed documents.',
    )
    parser.add_argument(
        '--batchsize',
        type=int,
        default=1,
        help='Nougat parser batch size.',
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=0,
        help='Dataloader workers for Nougat parser.',
    )
    parser.add_argument(
        'mmd_out',
        type=str,
        default=None,
        help='Directory to write mmd files to.',
    )
    args = parser.parse_args()

    _load_env_file(_find_env_file(Path.cwd()))

    checkpoint_raw = args.checkpoint or os.getenv('NOUGAT_CHECKPOINT')
    if not checkpoint_raw:
        print(
            '[error] Missing checkpoint. Pass --checkpoint or set NOUGAT_CHECKPOINT '
            'in environment/.adaparse.env.',
            file=sys.stderr,
        )
        return 2

    checkpoint = Path(checkpoint_raw).expanduser().resolve()
    if not checkpoint.exists():
        print(f'[error] Checkpoint path does not exist: {checkpoint}', file=sys.stderr)
        return 2

    try:
        pdfs = _collect_pdfs(args.pdf, args.pdf_dir, args.max_pdfs)
    except Exception as exc:
        print(f'[error] {exc}', file=sys.stderr)
        return 2

    log_dir = Path(args.log_dir).expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = Path(args.out_jsonl).expanduser().resolve()
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    print(f'[info] checkpoint: {checkpoint}')
    print(f'[info] parsing {len(pdfs)} pdf(s):')
    for p in pdfs:
        print(f'  - {p}')

    parser_config = NougatParserConfig(
        checkpoint=checkpoint,
        batchsize=max(1, args.batchsize),
        num_workers=max(0, args.num_workers),
        nougat_logs_path=log_dir,
        mmd_out=args.mmd_out,
    )
    parser_instance = NougatParser(parser_config)
    docs = parser_instance.parse([str(p) for p in pdfs])

    if not docs:
        print('[error] parser returned no documents.', file=sys.stderr)
        return 1

    with out_jsonl.open('w', encoding='utf-8') as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')

    non_empty = sum(1 for d in docs if (d.get('text') or '').strip())
    print(f'[ok] parsed documents: {len(docs)}')
    print(f'[ok] non-empty documents: {non_empty}')
    print(f'[ok] wrote jsonl: {out_jsonl}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
