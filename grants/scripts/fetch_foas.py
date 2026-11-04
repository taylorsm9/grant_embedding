"""Fetch NIH Guide pages for a list of FOA numbers (Phase 3, steps 1-2).

Usage: python -I fetch_foas.py <grants_dir> <foa_list.txt>

One FOA number per line. Saves raw/nih_guide/<FOA>.html and appends one row per
attempt to logs/foa_fetch.csv. FOAs already saved, or already recorded as
not_found, are skipped, so re-running resumes. RFA numbers are tried under
rfa-files/ first and everything else under pa-files/ first; the other directory
is the fallback.
"""
import csv
import logging
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

GUIDE = "https://grants.nih.gov/grants/guide"
# Requests identify the person running them; set GRANT_RETRIEVAL_CONTACT to your email.
CONTACT = os.environ.get("GRANT_RETRIEVAL_CONTACT", "")
UA = f"grant-retrieval-research/0.1 (contact: {CONTACT})"


def require_contact():
    if "@" not in CONTACT:
        sys.exit("Set GRANT_RETRIEVAL_CONTACT to a contact email; it is sent in the User-Agent header.")
LOG_FIELDS = ["foa_number", "status", "http_code", "source_url", "bytes", "fetched_at"]


class Blocked(Exception):
    pass


def get(url, attempts=4):
    """Return (http_code, body); body is None for 404/410."""
    for i in range(attempts):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            if e.code in (403, 429):
                raise Blocked(f"{e.code} from {url}")
            if e.code in (404, 410):
                return e.code, None
            if e.code < 500 or i == attempts - 1:
                return e.code, None
        except (urllib.error.URLError, TimeoutError):
            if i == attempts - 1:
                return 0, None
        time.sleep(2 ** (i + 1))


def main():
    require_contact()
    grants, list_path = Path(sys.argv[1]), Path(sys.argv[2])
    out = grants / "raw" / "nih_guide"
    out.mkdir(parents=True, exist_ok=True)
    log_path = grants / "logs" / "foa_fetch.csv"
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(grants / "logs" / "fetch_foas.log"), logging.StreamHandler()],
    )
    done = set()
    if log_path.exists():
        with open(log_path, newline="") as f:
            done = {r["foa_number"] for r in csv.DictReader(f) if r["status"] in ("ok", "not_found")}
    foas = [x.strip().upper() for x in list_path.read_text().split() if x.strip()]
    new_log = not log_path.exists()
    with open(log_path, "a", newline="") as lf:
        w = csv.DictWriter(lf, fieldnames=LOG_FIELDS)
        if new_log:
            w.writeheader()
        for n, foa in enumerate(foas, 1):
            if foa in done or (out / f"{foa}.html").exists():
                continue
            dirs = ["rfa-files", "pa-files"] if foa.startswith("RFA-") else ["pa-files", "rfa-files"]
            row = {"foa_number": foa, "status": "not_found", "http_code": "", "source_url": "", "bytes": 0}
            for d in dirs:
                url = f"{GUIDE}/{d}/{foa}.html"
                code, body = get(url)
                time.sleep(1)
                row.update(http_code=code, source_url=url)
                if body:
                    (out / f"{foa}.html").write_bytes(body)
                    row.update(status="ok", bytes=len(body))
                    break
                if code not in (404, 410):
                    row["status"] = "error"
            row["fetched_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            w.writerow(row)
            lf.flush()
            if n % 50 == 0:
                logging.info("%d/%d processed", n, len(foas))


if __name__ == "__main__":
    try:
        main()
    except Blocked as e:
        logging.error("STOP: server returned a blocking response: %s", e)
        sys.exit(2)
