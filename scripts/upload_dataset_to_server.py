"""
Sync local dataset files to the deployed SiameseScribe server.

Calls the /admin/dataset/* endpoints on the backend to:
  1. Show what's currently in the server's data volume (--info only)
  2. Diff local files against the server inventory
  3. Upload only files that are missing or have a different byte size

Usage:
    python scripts/upload_dataset_to_server.py \\
        --server-url https://your-server.example.com \\
        --admin-password secret \\
        [--admin-user admin] \\
        [--data-dir ./data] \\
        [--dirs patches/train patches/test dataset/preprocessed/train dataset/preprocessed/test] \\
        [--concurrency 10] \\
        [--dry-run] \\
        [--info]

Requirements:
    pip install httpx
"""

import argparse
import asyncio
import sys
from pathlib import Path

try:
    import httpx
except ImportError:
    print("httpx is required:  pip install httpx", file=sys.stderr)
    sys.exit(1)

_BASE_PATH = "/api"

AVAILABLE_DIRS = [
    "patches/train",
    "patches/test",
    "dataset/preprocessed/train",
    "dataset/preprocessed/test",
]


# CLI

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sync local dataset files to the SiameseScribe server.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--server-url", required=True,
                   help="Base URL of the deployed app, e.g. https://example.com")
    p.add_argument("--admin-user", default="admin", help="Admin username (default: admin)")
    p.add_argument("--admin-password", required=True, help="Admin password")
    p.add_argument("--data-dir", default="./data",
                   help="Local path to the data/ directory (default: ./data)")
    p.add_argument("--dirs", nargs="+", default=AVAILABLE_DIRS,
                   choices=AVAILABLE_DIRS, metavar="DIR",
                   help="Subdirectories to sync (default: all four)")
    p.add_argument("--concurrency", type=int, default=10,
                   help="Max concurrent uploads (default: 10)")
    p.add_argument("--dry-run", action="store_true",
                   help="Show what would be uploaded without sending any files")
    p.add_argument("--info", action="store_true",
                   help="Only show server directory info, then exit")
    p.add_argument("--count-files", action="store_true",
                   help="Include file counts in --info output (slow on large directories)")
    p.add_argument("--base-path", default="/api",
                   help="API path prefix (default: /api). Use empty string when hitting the "
                        "backend dev server directly on port 8000 (no nginx).")
    return p.parse_args()


# API helpers

def _url(server_url: str, path: str) -> str:
    return f"{server_url.rstrip('/')}{_BASE_PATH}{path}"


async def login(client: httpx.AsyncClient, server_url: str, user: str, password: str) -> str:
    resp = await client.post(
        _url(server_url, "/auth/login"),
        json={"username": user, "password": password},
    )
    if resp.status_code == 401:
        print("Login failed: wrong username or password.", file=sys.stderr)
        sys.exit(1)
    resp.raise_for_status()
    return resp.json()["access_token"]


async def get_info(client: httpx.AsyncClient, server_url: str, token: str,
                   count_files: bool = False) -> dict:
    resp = await client.get(
        _url(server_url, "/admin/dataset/info"),
        headers={"Authorization": f"Bearer {token}"},
        params={"count_files": "true" if count_files else "false"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


async def get_inventory(client: httpx.AsyncClient, server_url: str, token: str) -> dict[str, int]:
    print("Fetching server inventory (may take a moment for large directories) ...")
    resp = await client.get(
        _url(server_url, "/admin/dataset/inventory"),
        headers={"Authorization": f"Bearer {token}"},
        timeout=300,
    )
    resp.raise_for_status()
    data = resp.json()
    print(f"Server has {data['total']} files in upload directories.")
    return data["files"]


# Upload
async def upload_one(
    client: httpx.AsyncClient,
    server_url: str,
    token: str,
    local_path: Path,
    relative_path: str,
    sem: asyncio.Semaphore,
    counter: list[int],
    total: int,
) -> bool:
    async with sem:
        idx = counter[0] + 1
        counter[0] = idx
        size_kb = local_path.stat().st_size / 1024
        print(f"  [{idx}/{total}] {relative_path} ({size_kb:.1f} KB)")
        try:
            with open(local_path, "rb") as f:
                resp = await client.post(
                    _url(server_url, "/admin/dataset/upload-file"),
                    headers={"Authorization": f"Bearer {token}"},
                    data={"relative_path": relative_path},
                    files={"file": (local_path.name, f, "application/octet-stream")},
                    timeout=300,
                )
            if resp.status_code not in (200, 201):
                print(f"    ERROR {resp.status_code}: {resp.text}", file=sys.stderr)
                return False
            return True
        except Exception as exc:
            print(f"    ERROR uploading {relative_path}: {exc}", file=sys.stderr)
            return False


# Main
async def main() -> None:
    global _BASE_PATH
    args = parse_args()
    _BASE_PATH = args.base_path
    data_dir = Path(args.data_dir).resolve()

    if not data_dir.is_dir():
        print(f"Data directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    async with httpx.AsyncClient(timeout=30) as client:
        #  Auth
        print(f"Authenticating with {args.server_url} ...")
        token = await login(client, args.server_url, args.admin_user, args.admin_password)
        print("Authenticated.\n")

        # Server info
        print("=== Server data directory ===")
        info = await get_info(client, args.server_url, token, count_files=args.count_files)
        print(f"data_root : {info['data_root']}")
        for d, stats in info["directories"].items():
            marker = "✓" if stats["exists"] else "✗"
            count = stats["file_count"]
            count_str = f"{count} file(s)" if count is not None else "(not counted)"
            print(f"  {marker} {d}: {count_str}")
        print()

        if args.info:
            return

        # Confirm before proceeding
        try:
            confirm = input("Proceed with upload diff? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            sys.exit(0)
        if confirm != "y":
            print("Aborted.")
            sys.exit(0)
        print()

        # Inventory diff
        server_files = await get_inventory(client, args.server_url, token)

        to_upload: list[tuple[Path, str]] = []
        skipped = 0

        for rel_dir in args.dirs:
            local_dir = data_dir / rel_dir
            if not local_dir.is_dir():
                print(f"Skipping {rel_dir} — not found locally at {local_dir}")
                continue
            for f in sorted(local_dir.rglob("*")):
                if not f.is_file():
                    continue
                rel_path = f.relative_to(data_dir).as_posix()
                server_size = server_files.get(rel_path)
                if server_size is not None and server_size == f.stat().st_size:
                    skipped += 1
                else:
                    to_upload.append((f, rel_path))

        print(f"\nFiles to upload  : {len(to_upload)}")
        print(f"Already on server: {skipped} (skipped)")

        if args.dry_run:
            if to_upload:
                print("\nFirst 20 files that would be uploaded:")
                for _, rel in to_upload[:20]:
                    print(f"  {rel}")
                if len(to_upload) > 20:
                    print(f"  ... and {len(to_upload) - 20} more")
            print("\n[dry-run] No files were uploaded.")
            return

        if not to_upload:
            print("Nothing to upload — server is already up to date.")
            return

        # Upload
        print(f"\nUploading {len(to_upload)} files with concurrency={args.concurrency} ...")
        sem = asyncio.Semaphore(args.concurrency)
        counter: list[int] = [0]
        total = len(to_upload)

        tasks = [
            upload_one(client, args.server_url, token, lp, rp, sem, counter, total)
            for lp, rp in to_upload
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        errors = sum(1 for r in results if r is not True)
        print(f"\nDone.  Uploaded: {total - errors}  Errors: {errors}")
        if errors:
            sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
