"""Download base models and voice models for inference-only RVC.

Usage:
    python tools/download_models.py --base hubert rmvpe
    python tools/download_models.py --url https://.../model.pth
    python tools/download_models.py --url https://.../model.zip

Set RVC_HF_ENDPOINT to override the Hugging Face endpoint (e.g. hf-mirror.com).
"""

import argparse
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from urllib.parse import unquote, urlparse

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent

HF_ENDPOINT = os.environ.get("RVC_HF_ENDPOINT", "https://huggingface.co").rstrip("/")
BASE_REPO = "lj1995/VoiceConversionWebUI"

WEIGHT_ROOT = PROJECT_ROOT / "assets" / "weights"
INDEX_ROOT = PROJECT_ROOT / "assets" / "indices"

# key -> list of (repo_path, destination relative to project root)
BASE_ASSETS = {
    "hubert": [
        ("hubert_base/config.json", "assets/hubert_base/config.json"),
        (
            "hubert_base/preprocessor_config.json",
            "assets/hubert_base/preprocessor_config.json",
        ),
        ("hubert_base/pytorch_model.bin", "assets/hubert_base/pytorch_model.bin"),
    ],
    "rmvpe": [("rmvpe.pt", "assets/rmvpe/rmvpe.pt")],
}

CHUNK_SIZE = 1024 * 1024


def _asset_url(repo_path):
    return "%s/%s/resolve/main/%s" % (HF_ENDPOINT, BASE_REPO, repo_path)


def _filename_from_url(url):
    path = unquote(urlparse(url).path)
    return os.path.basename(path) or "download.bin"


def download_file(url, dest, progress=None, min_bytes=1):
    """Stream ``url`` to ``dest`` atomically. Returns the destination path."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(dir=str(dest.parent), suffix=".part")
    os.close(tmp_fd)
    tmp_path = Path(tmp_name)
    try:
        with requests.get(url, stream=True, timeout=30, allow_redirects=True) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("Content-Length") or 0)
            done = 0
            with open(tmp_path, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        fh.write(chunk)
                        done += len(chunk)
                        if progress is not None:
                            progress(done, total, url)
        if not tmp_path.exists() or tmp_path.stat().st_size < min_bytes:
            raise RuntimeError("Downloaded file is empty or truncated: %s" % url)
        shutil.move(str(tmp_path), str(dest))
    finally:
        if tmp_path.exists():
            try:
                os.remove(tmp_path)
            except OSError:
                pass
    return dest


def download_base_assets(keys=None, force=False, progress=None):
    """Download required inference base models. Returns status message list."""
    keys = list(keys) if keys else ["hubert", "rmvpe"]
    unknown = [key for key in keys if key not in BASE_ASSETS]
    if unknown:
        raise ValueError(
            "Unknown base asset(s): %s (choose from %s)"
            % (", ".join(unknown), ", ".join(sorted(BASE_ASSETS)))
        )
    messages = []
    for key in keys:
        for repo_path, rel_dest in BASE_ASSETS[key]:
            dest = PROJECT_ROOT / rel_dest
            if dest.is_file() and dest.stat().st_size > 0 and not force:
                messages.append("[skip] %s sudah ada" % rel_dest)
                continue
            url = _asset_url(repo_path)
            print("Downloading %s -> %s" % (url, rel_dest), flush=True)
            download_file(url, dest, progress=progress)
            size_mb = dest.stat().st_size / (1024 * 1024)
            messages.append("[ok] %s (%.1f MB)" % (rel_dest, size_mb))
    return messages


def download_voice_model_url(
    url,
    name=None,
    force=False,
    progress=None,
    weight_root=WEIGHT_ROOT,
    index_root=INDEX_ROOT,
):
    """Download a .pth/.index/.zip voice model from ``url``.

    Returns a status message list. .pth files go to ``weight_root``, .index
    files to ``index_root``; zip archives are unpacked and sorted by extension.
    """
    weight_root = Path(weight_root)
    index_root = Path(index_root)
    clean_url = urlparse(url)
    filename = name or _filename_from_url(url)
    lower = filename.lower()

    if clean_url.scheme not in ("http", "https"):
        raise ValueError("URL harus http/https: %s" % url)

    saved = []

    def save(data, target):
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file() and not force:
            saved.append("[skip] %s sudah ada" % target)
            return
        with open(target, "wb") as fh:
            fh.write(data)
        size_mb = target.stat().st_size / (1024 * 1024)
        saved.append("[ok] %s (%.1f MB)" % (target, size_mb))

    if lower.endswith(".zip"):
        with tempfile.TemporaryDirectory() as tmp_dir:
            zip_path = download_file(
                url, Path(tmp_dir) / filename, progress=progress
            )
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(tmp_dir)
            for root, _, files in os.walk(tmp_dir):
                for file_name in files:
                    src = Path(root) / file_name
                    if src == zip_path or file_name.lower().endswith(".zip"):
                        continue
                    if file_name.lower().endswith(".pth"):
                        save(src.read_bytes(), weight_root / file_name)
                    elif file_name.lower().endswith(".index"):
                        save(src.read_bytes(), index_root / file_name)
    elif lower.endswith(".pth"):
        data = _fetch_bytes(url, progress)
        save(data, weight_root / filename)
    elif lower.endswith(".index"):
        data = _fetch_bytes(url, progress)
        save(data, index_root / filename)
    else:
        raise ValueError(
            "Ekstensi tidak dikenal untuk '%s' (.pth/.index/.zip)" % filename
        )
    return saved


def _fetch_bytes(url, progress=None):
    import io

    buffer = io.BytesIO()
    with requests.get(url, stream=True, timeout=30, allow_redirects=True) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
            if chunk:
                buffer.write(chunk)
                done += len(chunk)
                if progress is not None:
                    progress(done, total, url)
    return buffer.getvalue()


def main():
    parser = argparse.ArgumentParser(description="RVC inference model downloader")
    parser.add_argument(
        "--base",
        nargs="*",
        choices=sorted(BASE_ASSETS),
        help="Base models to download (default: all when flag is used without values)",
    )
    parser.add_argument("--url", action="append", default=[], help="Voice model URL")
    parser.add_argument("--force", action="store_true", help="Re-download existing files")
    args = parser.parse_args()

    def cli_progress(done, total, url):
        name = _filename_from_url(url)
        if total:
            print("\r%s: %.1f%%" % (name, done / total * 100), end="", flush=True)
        else:
            print("\r%s: %.1f MB" % (name, done / (1024 * 1024)), end="", flush=True)

    try:
        if args.base is not None:
            keys = args.base or sorted(BASE_ASSETS)
            for line in download_base_assets(keys, args.force, cli_progress):
                print(line, flush=True)
        for url in args.url:
            for line in download_voice_model_url(url, force=args.force, progress=cli_progress):
                print(line, flush=True)
        if args.base is None and not args.url:
            parser.print_help()
    except Exception as error:
        print("ERROR: %s" % error, file=sys.stderr, flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
