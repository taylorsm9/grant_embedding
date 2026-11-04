"""Download NIH ExPORTER project and abstract zips (Phase 1, steps 1-2).

Usage: python -I download_exporter.py <grants_dir> [first_fy] [last_fy]

Years download newest first; with no range given, every listed year is fetched.

Lists files via the RePORTER ExPORTER service, downloads each FY zip into
raw/nih_exporter/{projects,abstracts}/, validates it (zip opens, CSV has a
header row), and appends to raw/nih_exporter/manifest.csv. Re-running skips
files already in the manifest whose size and SHA-256 still match.
"""
import csv
import hashlib
import io
import json
import logging
import os
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

BASE = "https://reporter.nih.gov/services/exporter"
# Requests identify the person running them; set GRANT_RETRIEVAL_CONTACT to your email.
CONTACT = os.environ.get("GRANT_RETRIEVAL_CONTACT", "")
UA = f"grant-retrieval-research/0.1 (contact: {CONTACT})"


def require_contact():
    if "@" not in CONTACT:
        sys.exit("Set GRANT_RETRIEVAL_CONTACT to a contact email; it is sent in the User-Agent header.")
GROUPS = {"PROJECT": "projects", "ABSTRACT": "abstracts"}
MANIFEST_FIELDS = ["group", "fy", "file_name", "size", "sha256", "source_url", "downloaded_at"]


class Blocked(Exception):
    pass


def request(url, data=None, attempts=5):
    headers = {"User-Agent": UA}
    if data is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(data).encode()
    for i in range(attempts):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, data=data, headers=headers), timeout=300) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (403, 429):
                raise Blocked(f"{e.code} from {url}")
            if e.code < 500 or i == attempts - 1:
                raise
        except (urllib.error.URLError, TimeoutError):
            if i == attempts - 1:
                raise
        wait = 2 ** (i + 1)
        logging.warning("retry %d for %s in %ds", i + 1, url, wait)
        time.sleep(wait)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_zip(path):
    with zipfile.ZipFile(path) as z:
        csvs = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if len(csvs) != 1:
            raise ValueError(f"expected one CSV in {path.name}, found {csvs}")
        with z.open(csvs[0]) as f:
            header = io.TextIOWrapper(f, encoding="utf-8-sig", errors="replace").readline()
    if "APPLICATION_ID" not in header.upper():
        raise ValueError(f"no APPLICATION_ID in header of {path.name}: {header[:200]!r}")


def load_manifest(path):
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        return {(r["group"], r["file_name"]): r for r in csv.DictReader(f)}


def main():
    require_contact()
    root = Path(sys.argv[1])
    first, last = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (0, 9999)
    raw = root / "raw" / "nih_exporter"
    (root / "logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(root / "logs" / "download_exporter.log"), logging.StreamHandler()],
    )
    manifest_path = raw / "manifest.csv"
    manifest = load_manifest(manifest_path)
    new_manifest = not manifest_path.exists()

    with open(manifest_path, "a", newline="") as mf:
        writer = csv.DictWriter(mf, fieldnames=MANIFEST_FIELDS)
        if new_manifest:
            writer.writeheader()
        for group, subdir in GROUPS.items():
            listing = json.loads(request(f"{BASE}/allFilesInfo", {"file_group": group}))
            (raw / "listings").mkdir(parents=True, exist_ok=True)
            (raw / "listings" / f"{group}.json").write_text(json.dumps(listing, indent=1))
            time.sleep(1)
            for item in sorted(listing, key=lambda x: x["fy"], reverse=True):
                if not (first <= item["fy"] <= last) or item.get("to_fy"):
                    continue
                dest = raw / subdir / item["file_name"]
                prev = manifest.get((group, item["file_name"]))
                if prev and dest.exists() and dest.stat().st_size == int(prev["size"]) and sha256(dest) == prev["sha256"]:
                    logging.info("skip %s (already valid)", dest.name)
                    continue
                url = f"{BASE}/DownloadFromDocService?DocType={item['doc_type_code']}&KeyId={item['doc_key_id']}"
                logging.info("download %s FY%d -> %s", group, item["fy"], dest.name)
                tmp = dest.with_suffix(".part")
                tmp.write_bytes(request(url))
                validate_zip(tmp)
                tmp.rename(dest)
                row = {
                    "group": group, "fy": item["fy"], "file_name": dest.name,
                    "size": dest.stat().st_size, "sha256": sha256(dest), "source_url": url,
                    "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
                writer.writerow(row)
                mf.flush()
                time.sleep(1)


if __name__ == "__main__":
    try:
        main()
    except Blocked as e:
        logging.error("STOP: server returned a blocking response: %s", e)
        sys.exit(2)
