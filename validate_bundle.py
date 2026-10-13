#!/usr/bin/env python3
"""Offline integrity checks. Never builds or launches an Ascend kernel."""
import argparse
import ast
import hashlib
import json
import os
import pathlib
import re
import subprocess
from urllib.parse import unquote, urlparse


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def contained(path, root):
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=pathlib.Path, help='Write a JSON report at this path')
    args = parser.parse_args()
    root = pathlib.Path(__file__).resolve().parent
    posts = sorted(p for p in root.iterdir() if p.is_dir() and re.match(r'^\d{2}-', p.name))
    errors = []
    counts = {
        'posts': len(posts), 'snapshot_files': 0, 'release_evidence_hashes': 0,
        'json_files': 0, 'python_ast_files': 0, 'shell_syntax_files': 0,
        'wrapper_help_commands': 0, 'authored_markdown_files': 0,
        'relative_links': 0, 'bundle_files': 0, 'ascend_c_posts': 0,
    }
    post_results = []
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')

    def check(condition, message):
        if not condition:
            errors.append(message)

    def execute(command, label):
        try:
            result = subprocess.run(command, cwd=str(root), env=env, capture_output=True,
                                    text=True, timeout=30)
            check(result.returncode == 0, label + ': ' + (result.stderr or result.stdout).strip()[:1200])
        except (OSError, subprocess.TimeoutExpired) as exc:
            errors.append(label + ': ' + str(exc))

    required = ['docs/post.md', 'docs/reproduce.md', 'docs/sources.md', 'docs/metadata.json',
                'code/README.md', 'code/reproduce.sh', 'code/source-manifest.json',
                'code/upstream/LICENSE']
    check(bool(posts), 'No post folders found')
    for post in posts:
        before = len(errors)
        for relative in required:
            check((post / relative).is_file(), post.name + ': missing ' + relative)
        try:
            metadata = json.loads((post / 'docs/metadata.json').read_text(encoding='utf-8'))
            manifest = json.loads((post / 'code/source-manifest.json').read_text(encoding='utf-8'))
            commit = manifest['commit']
            metadata_commit = metadata.get('commit', metadata.get('upstream', {}).get('commit'))
            check(re.fullmatch(r'[0-9a-f]{40}', commit) is not None, post.name + ': invalid commit')
            check(metadata_commit == commit, post.name + ': metadata/manifest commit mismatch')
            check(metadata.get('implementation_language') == 'Ascend C', post.name + ': missing Ascend C classification')
            for entry in manifest['files']:
                relative = pathlib.Path(entry['path'])
                base = post / 'code' if relative.parts[0] == 'upstream' else post / 'code/upstream'
                path = base / relative
                check(contained(path, post / 'code/upstream'), post.name + ': unsafe snapshot path')
                if not contained(path, post / 'code/upstream') or not path.is_file():
                    errors.append(post.name + ': missing snapshot ' + str(relative))
                    continue
                check(digest(path) == entry['sha256'], post.name + ': hash mismatch ' + str(relative))
                check(path.stat().st_size == entry['bytes'], post.name + ': size mismatch ' + str(relative))
                url = entry.get('source_url', entry.get('url', ''))
                check(commit in url, post.name + ': unpinned source URL ' + str(relative))
                counts['snapshot_files'] += 1
            for entry in manifest.get('evidence_sources', []):
                path = post / 'code' / entry['local_path']
                check(contained(path, post), post.name + ': unsafe evidence path')
                if path.is_file() and contained(path, post):
                    check(digest(path) == entry['sha256'], post.name + ': release evidence hash mismatch')
                    counts['release_evidence_hashes'] += 1
                else:
                    errors.append(post.name + ': missing release evidence')
            candidates = [f for f in (post / 'code').rglob('*')
                          if f.suffix in ('.cpp', '.h', '.hpp', '.py') and f.is_file()]
            check(bool(candidates), post.name + ': no actual implementation source')
            kernels = [f for f in candidates if f.suffix in ('.h', '.hpp', '.cpp') and 'op_kernel' in f.parts]
            ascend_c = any('AscendC' in f.read_text(encoding='utf-8') or 'kernel_operator.h' in f.read_text(encoding='utf-8')
                           for f in kernels)
            check(ascend_c, post.name + ': missing Ascend C kernel evidence')
            counts['ascend_c_posts'] += int(ascend_c)
            article = (post / 'docs/post.md').read_text(encoding='utf-8')
            check(len(article) > 2000, post.name + ': article is unexpectedly short')
            post_results.append({'id': post.name, 'snapshot_files': len(manifest['files']),
                                 'commit': commit, 'passed': len(errors) == before})
        except (OSError, ValueError, KeyError, IndexError) as exc:
            errors.append(post.name + ': ' + str(exc))
        if (post / 'code/reproduce.sh').is_file():
            execute(['bash', str(post / 'code/reproduce.sh'), '--help'], post.name + ' --help')
            counts['wrapper_help_commands'] += 1

    for path in sorted(root.rglob('*')):
        if not path.is_file():
            continue
        relative = str(path.relative_to(root))
        check(not any(part in ('__pycache__', '.git') for part in path.parts), 'Unwanted cache/repository: ' + relative)
        check(path.suffix != '.pyc' and path.name != '.DS_Store', 'Unwanted generated file: ' + relative)
        if path.suffix == '.json':
            try:
                json.loads(path.read_text(encoding='utf-8'))
                counts['json_files'] += 1
            except (OSError, ValueError) as exc:
                errors.append(relative + ': invalid JSON: ' + str(exc))
        if path.suffix == '.py':
            try:
                ast.parse(path.read_text(encoding='utf-8-sig'), filename=relative)
                counts['python_ast_files'] += 1
            except (OSError, SyntaxError, UnicodeError) as exc:
                errors.append(relative + ': invalid Python AST: ' + str(exc))
        if path.suffix == '.sh':
            execute(['bash', '-n', str(path)], relative + ' bash -n')
            counts['shell_syntax_files'] += 1
        if path.suffix == '.md' and 'upstream' not in path.parts and 'evidence' not in path.parts:
            counts['authored_markdown_files'] += 1
            body = re.sub(r'(?ms)^```.*?^```[^\n]*', '', path.read_text(encoding='utf-8'))
            for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)', body):
                target = target.strip()
                if target.startswith('<'):
                    target = target[1:].split('>', 1)[0]
                else:
                    target = target.split()[0]
                parsed = urlparse(target)
                if parsed.scheme or parsed.netloc or not parsed.path:
                    continue
                link_path = pathlib.Path(unquote(parsed.path))
                check(not link_path.is_absolute(), relative + ': nonportable absolute file link ' + target)
                linked = path.parent / link_path
                check(contained(linked, root) and linked.exists(), relative + ': broken relative link ' + target)
                counts['relative_links'] += 1

    try:
        catalog = json.loads((root / 'catalog.json').read_text(encoding='utf-8'))
        check(catalog['post_count'] == len(posts), 'catalog post count mismatch')
        check(catalog['snapshot_file_entries'] == counts['snapshot_files'], 'catalog source count mismatch')
        check({p['id'] for p in catalog['posts']} == {p.name for p in posts}, 'catalog IDs mismatch')
        for item in catalog['posts']:
            check(item.get('implementation', {}).get('language') == 'Ascend C', 'catalog contains non-Ascend C post: ' + item['id'])
            for path in item.get('implementation', {}).get('evidence_paths', []):
                check((root / path).is_file(), 'catalog missing Ascend C evidence: ' + path)
            for path in item['paths'].values():
                check((root / path).exists(), 'catalog missing path: ' + path)
    except (OSError, ValueError, KeyError) as exc:
        errors.append('catalog: ' + str(exc))

    bundle = root / 'bundle-manifest.json'
    bundle_status = 'not_present'
    if bundle.is_file():
        bundle_status = 'checked'
        try:
            manifest = json.loads(bundle.read_text(encoding='utf-8'))
            expected = {item['path'] for item in manifest['files']}
            actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p != bundle}
            check(expected == actual, 'bundle file inventory mismatch')
            for item in manifest['files']:
                path = root / item['path']
                check(contained(path, root) and path.is_file(), 'missing/unsafe bundle path: ' + item['path'])
                if path.is_file() and contained(path, root):
                    check(digest(path) == item['sha256'], 'bundle hash mismatch: ' + item['path'])
                    check(path.stat().st_size == item['bytes'], 'bundle size mismatch: ' + item['path'])
                    counts['bundle_files'] += 1
        except (OSError, ValueError, KeyError) as exc:
            errors.append('bundle manifest: ' + str(exc))

    report = {'passed': not errors, 'counts': counts, 'bundle_manifest': bundle_status,
              'npu_compile': 'not_run', 'npu_accuracy': 'not_run', 'npu_performance': 'not_run',
              'posts': post_results, 'errors': errors}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'passed': report['passed'], 'counts': counts, 'bundle_manifest': bundle_status,
                      'errors': errors}, ensure_ascii=False, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
