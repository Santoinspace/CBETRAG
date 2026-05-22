"""Task 1: Data adapter — unifies AdaRAGUE CSV formats into Question dataclass."""
from __future__ import annotations
import re
import ast
from dataclasses import dataclass
from pathlib import Path
import pandas as pd


@dataclass
class Question:
    qid: str
    query: str
    gold_passages: list[str]
    distractor_passages: list[str]
    answer: str
    dataset: str
    hop_count: int


def _parse_context(s: str) -> list[dict]:
    """Parse AdaRAGUE context field (numpy repr with \r\n between dicts)."""
    s = re.sub(r'array\((\[.*?\]),\s*dtype=object\)', r'\1', s, flags=re.DOTALL)
    s = re.sub(r'\}\s*\{', '}, {', s)
    return eval(s, {'__builtins__': {}})  # noqa: S307 — trusted internal data


def _parse_answer(s: str) -> str:
    """Extract first span from answer_objects field."""
    s = re.sub(r'array\((\[.*?\]),\s*dtype=object\)', r'\1', s, flags=re.DOTALL)
    objs = eval(s, {'__builtins__': {}})  # noqa: S307
    spans = objs[0].get('spans', [])
    return spans[0] if spans else ''


def _hop_count_musique(qid: str) -> int:
    m = re.match(r'^(\d+)hop', qid)
    return int(m.group(1)) if m else 2


def _hop_count_2wiki(qid: str, qtype: str = '') -> int:
    if 'bridge_comparison' in qtype:
        return 3
    return 2


def _row_to_question(row: pd.Series, dataset: str) -> Question:
    ctx = _parse_context(row['context'])
    gold = [c['paragraph_text'] for c in ctx if c.get('is_supporting')]
    distractor = [c['paragraph_text'] for c in ctx if not c.get('is_supporting')]
    answer = _parse_answer(row['answer_objects'])

    if dataset == 'musique':
        hop = _hop_count_musique(str(row['qid']))
    elif dataset == '2wikimultihopqa':
        hop = _hop_count_2wiki(str(row['qid']))
    else:  # hotpotqa
        hop = 2

    return Question(
        qid=str(row['qid']),
        query=str(row['question_text']),
        gold_passages=gold,
        distractor_passages=distractor,
        answer=answer,
        dataset=dataset,
        hop_count=hop,
    )


def load_dataset(dataset: str, split: str = 'test',
                 n_samples: int | None = None,
                 data_root: str = 'AdaRAGUE/data') -> list[Question]:
    """Load AdaRAGUE dataset CSV and return list of Question objects.

    Args:
        dataset: 'hotpotqa' | 'musique' | '2wikimultihopqa'
        split: 'test' or 'train'
        n_samples: if set, return only first n rows
        data_root: path to AdaRAGUE/data directory
    """
    path = Path(data_root) / f'adaptive_rag_{dataset}' / f'{split}.csv'
    df = pd.read_csv(path, nrows=n_samples)
    return [_row_to_question(row, dataset) for _, row in df.iterrows()]
