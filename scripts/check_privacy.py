#!/usr/bin/env python
"""Reject personal machine details and credential patterns without echoing matches."""
import argparse
import re
import subprocess
from pathlib import Path

PATTERNS = {
    'personal filesystem path': re.compile(r'/(?:Users|home|root|private/tmp|tmp)/[^\s`\"\'<>]+'),
    'Windows user path': re.compile(r'[A-Za-z]:[\\/]+Users[\\/]+[^\s\"\']+', re.I),
    'internal UUID': re.compile(r'\b[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}\b', re.I),
    'local hostname': re.compile(r'\b[A-Za-z0-9][A-Za-z0-9.-]*\.local\b', re.I),
    'private address': re.compile(r'\b(?:10|192\.168|172\.(?:1[6-9]|2[0-9]|3[01]))(?:\.\d{1,3}){2,3}\b'),
    'credential': re.compile(r'(?:gh[pousr]_|github_pat_|sk-(?:proj-|svcacct-)?)[A-Za-z0-9_-]{16,}'),
    'AWS access key': re.compile(r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'),
    'private key': re.compile('-----BEGIN ' + r'(?:[A-Z]+ )?PRIVATE KEY-----'),
    'personal email': re.compile(r'\b[A-Za-z0-9_.+-]+@(?!users\.noreply\.github\.com\b)[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'),
}


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args])


def check(text, label):
    failures = []
    for category, pattern in PATTERNS.items():
        for match in pattern.finditer(text):
            line = text.count('\n', 0, match.start()) + 1
            failures.append(f'{label}:{line}: {category}')
    if failures:
        raise ValueError('Privacy check failed (values withheld):\n' + '\n'.join(failures))


def scan_worktree(root):
    names = set(git(root, 'ls-files', '--cached', '--others', '--exclude-standard', '-z').decode().split('\0'))
    for name in sorted(names - {''}):
        check(name, 'filename')
        path = root / name
        if path.is_file():
            try:
                body = path.read_text(encoding='utf-8')
            except UnicodeDecodeError:
                continue
            check(body, name)


def scan_history(root):
    commits = git(root, 'rev-list', '--all').decode().splitlines()
    for commit in commits:
        check(git(root, 'show', '-s', '--format=%an <%ae>%n%cn <%ce>%n%B', commit).decode(), commit[:12])
        names = git(root, 'ls-tree', '-r', '--name-only', '-z', commit).decode().split('\0')
        for name in filter(None, names):
            check(name, 'filename')
            try:
                body = git(root, 'show', f'{commit}:{name}').decode('utf-8')
            except UnicodeDecodeError:
                continue
            check(body, f'{commit[:12]}/{name}')
    return len(commits)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--history', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    scan_worktree(root)
    count = scan_history(root) if args.history else 0
    print(f'privacy_verified: true; history_commits: {count}')
