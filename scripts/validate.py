#!/usr/bin/env python
"""Validate the day's three lessons against their pinned local source."""
import argparse
import datetime
import json
import os
import re
import subprocess
from pathlib import Path

from check_privacy import scan_worktree

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get('VLLM_SOURCE_DIR', str(ROOT.parent / 'vllm')))


def validate(date):
    scan_worktree(ROOT)
    datetime.date.fromisoformat(date)
    index = json.loads((ROOT / 'index.json').read_text())
    topics = [lesson['topic_id'] for day in index['days'] for lesson in day['lessons']]
    if len(topics) != len(set(topics)):
        raise ValueError('index.json contains duplicate topic_id values')
    dates = [day['date'] for day in index['days']]
    if len(dates) != len(set(dates)):
        raise ValueError('index.json contains duplicate dates')
    day = next((item for item in index['days'] if item['date'] == date), None)
    if day is None or len(day['lessons']) != 3:
        raise ValueError('The requested day must contain exactly three indexed lessons')
    expected = set()
    for lesson in day['lessons']:
        relative = Path(lesson['file'])
        path = (ROOT / relative).resolve()
        if path.parent != ROOT / 'daily' / date or path.suffix != '.md':
            raise ValueError('Lesson path must be directly under daily/YYYY-MM-DD/')
        expected.add(path)
        body = path.read_text()
        if lesson['status'] != 'complete' or len(body) < 1200:
            raise ValueError(f'{relative}: incomplete lesson or insufficient body')
        if lesson['evidence'] != 'static_source_review':
            raise ValueError(f'{relative}: unsupported evidence; review validator first')
        if '静态源码' not in body or '教学' not in body:
            raise ValueError(f'{relative}: missing explicit evidence boundary')
        if body.count('```') % 2:
            raise ValueError(f'{relative}: unclosed code fence')
        commit = lesson['source_commit']
        if not re.fullmatch(r'[0-9a-f]{40}', commit) or commit not in body:
            raise ValueError(f'{relative}: missing full source commit')
        if not lesson['sources'] or not lesson['evidence']:
            raise ValueError(f'{relative}: missing evidence or sources')
        for source in lesson['sources']:
            lines = subprocess.check_output(
                ['git', '-C', str(SOURCE), 'show', f"{commit}:{source['path']}"],
                text=True,
            ).splitlines()
            line = source['line']
            if line < 1 or line > len(lines) or source['symbol'] not in lines[line - 1]:
                raise ValueError(f'{relative}: source anchor disagrees with pinned commit')
            url = f"https://github.com/vllm-project/vllm/blob/{commit}/{source['path']}#L{line}"
            if url not in body:
                raise ValueError(f'{relative}: missing pinned source link {url}')
        links = set(re.findall(
            r'https://github\.com/vllm-project/vllm/blob/'
            r'[0-9a-f]{40}/[^\s)]+', body,
        ))
        indexed_links = {
            f"https://github.com/vllm-project/vllm/blob/{commit}/"
            f"{source['path']}#L{source['line']}"
            for source in lesson['sources']
        }
        if links != indexed_links:
            raise ValueError(f'{relative}: source links and indexed anchors differ')
        for local_image in re.findall(r'!\[[^\]]*\]\(([^)]+)\)', body):
            if not local_image.startswith('https://'):
                image = (path.parent / local_image).resolve()
                if not image.is_relative_to(ROOT) or not image.is_file():
                    raise ValueError(f'{relative}: missing or unsafe local image')
        if lesson['file'] not in (ROOT / 'README.md').read_text():
            raise ValueError(f'{relative}: missing README archive entry')
    actual = set((ROOT / 'daily' / date).glob('*.md'))
    if len(expected) != 3 or actual != expected:
        raise ValueError('Daily directory and index must contain the same three files')
    return day


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('date')
    args = parser.parse_args()
    validate(args.date)
    print(f'{args.date}: three lessons and pinned source anchors verified (static evidence only)')
