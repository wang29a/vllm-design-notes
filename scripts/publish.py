#!/usr/bin/env python
"""Publish only the requested day's lessons and verify GitHub main."""
import argparse
import fcntl
import re
import subprocess
from contextlib import contextmanager

from validate import ROOT, validate
from check_privacy import scan_history



def run(*args, capture=False):
    result = subprocess.run(args, cwd=ROOT, check=True, text=True,
                            stdout=subprocess.PIPE if capture else None)
    return result.stdout.strip() if capture else None


@contextmanager
def lock():
    with (ROOT / '.publish.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def publish(date):
    day = validate(date)
    if run('git', 'branch', '--show-current', capture=True) != 'main':
        raise ValueError('Publish from main only')
    remote = run('git', 'remote', 'get-url', 'origin', capture=True)
    match = re.fullmatch(r'(?:https://github\.com/|git@github\.com:)([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+?)(?:\.git)?', remote)
    if match is None:
        raise ValueError('Origin must be a GitHub repository without embedded credentials')
    repository = match.group(1)
    scan_history(ROOT)
    files = ['README.md', 'index.json'] + [lesson['file'] for lesson in day['lessons']]
    staged = run('git', 'diff', '--cached', '--name-only', capture=True).splitlines()
    if set(staged) - set(files):
        raise ValueError('Unrelated staged files: preserve them and publish separately')
    tracked = run('git', 'diff', '--name-only', capture=True).splitlines()
    untracked = run('git', 'ls-files', '--others', '--exclude-standard', capture=True).splitlines()
    if (set(tracked) | set(untracked)) - set(files):
        raise ValueError('Unrelated working changes: preserve them and resolve before publishing')
    run('gh', 'api', f'repos/{repository}', '--jq', '.full_name')
    run('git', 'add', '--', *files)
    run('git', 'diff', '--cached', '--check')
    changed = subprocess.run(['git', 'diff', '--cached', '--quiet'], cwd=ROOT).returncode
    if changed == 1:
        run('git', 'commit', '-m', f'Learn three vLLM designs for {date}',
            '-m', 'Co-authored-by: Codex <codex@users.noreply.github.com>\n'
            'Signed-off-by: Design Notes Contributors '
            '<contributors@users.noreply.github.com>')
    elif changed != 0:
        raise RuntimeError('Unable to inspect staged diff')
    scan_history(ROOT)
    # Use existing gh credentials without printing or copying tokens.
    run('git', '-c', 'credential.helper=', '-c', 'credential.helper=!gh auth git-credential',
        'push', 'origin', 'main')
    local = run('git', 'rev-parse', 'HEAD', capture=True)
    remote_head = run('gh', 'api', f'repos/{repository}/git/ref/heads/main',
                      '--jq', '.object.sha', capture=True)
    if local != remote_head:
        raise RuntimeError('Push not verified: GitHub main differs from local HEAD')
    print(f'github_verified: true; date: {date}; commit: {local}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('date')
    args = parser.parse_args()
    with lock():
        publish(args.date)
