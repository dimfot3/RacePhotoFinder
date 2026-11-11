"""Download the InsightFace buffalo_l face models from their official release.

The weights are fetched directly from InsightFace, not from this repository,
and are verified against the SHA-256 checksums in THIRD_PARTY_NOTICES.md.
They are licensed for non-commercial research only.
"""
import argparse
import hashlib
from pathlib import Path
import sys
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent
URL = 'https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip'
TERMS = 'https://github.com/deepinsight/insightface#license'
MODELS = {
    'det_10g.onnx': (ROOT / 'face_detection/weights/det_10g.onnx',
                     '5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91'),
    'w600k_r50.onnx': (ROOT / 'face_matching/weights/w600k_r50.onnx',
                       '4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43'),
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as file:
        for chunk in iter(lambda: file.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def is_installed(path, checksum):
    return path.is_file() and sha256(path) == checksum


def confirm_terms(accepted):
    print('InsightFace buffalo_l models are available for non-commercial research only.')
    print(f'Review the terms before continuing: {TERMS}')
    if accepted:
        return True
    if not sys.stdin.isatty():
        print('Run with --accept-license to confirm non-interactively.')
        return False
    return input('Do you accept these terms? [y/N] ').strip().lower() in ('y', 'yes')


def download(url, destination):
    print(f'Downloading {url}')
    with urllib.request.urlopen(url) as response, open(destination, 'wb') as file:
        total = int(response.headers.get('Content-Length', 0))
        done = 0
        for chunk in iter(lambda: response.read(1 << 20), b''):
            file.write(chunk)
            done += len(chunk)
            if total:
                print(f'\r  {done / 1e6:.0f} / {total / 1e6:.0f} MB', end='', flush=True)
    print()


def extract(archive):
    with zipfile.ZipFile(archive) as zf:
        members = {Path(name).name: name for name in zf.namelist()}
        for name, (path, checksum) in MODELS.items():
            if name not in members:
                raise RuntimeError(f'{name} is missing from the downloaded archive')
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_suffix('.part')
            with zf.open(members[name]) as source, open(partial, 'wb') as target:
                while chunk := source.read(1 << 20):
                    target.write(chunk)
            if sha256(partial) != checksum:
                partial.unlink()
                raise RuntimeError(f'Checksum mismatch for {name}; the file was not installed')
            partial.replace(path)
            print(f'Installed {path.relative_to(ROOT)} (SHA-256 verified)')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--accept-license', action='store_true',
                        help='Confirm the InsightFace non-commercial terms without a prompt')
    args = parser.parse_args()

    if all(is_installed(path, checksum) for path, checksum in MODELS.values()):
        print('Face models are already installed and verified.')
        return 0
    if not confirm_terms(args.accept_license):
        print('Cancelled. No files were downloaded.')
        return 1
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / 'buffalo_l.zip'
        download(URL, archive)
        extract(archive)
    print('Face models are ready.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
