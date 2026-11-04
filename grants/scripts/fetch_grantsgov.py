"""Grants.gov fallback for FOAs whose NIH Guide page was not found (master decision 6).

Usage: python -I fetch_grantsgov.py <grants_dir> <foa_list.txt>

For each FOA number: search2 by exact opportunity number (all statuses), then
fetchOpportunity for the matching id. Saves raw/grantsgov/<FOA>.json and appends to
logs/grantsgov_fetch.csv. Already-saved FOAs are skipped. About 1 request per second;
stops on 403/429.
"""
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

API = "https://api.grants.gov/v1/api"
# Requests identify the person running them; set GRANT_RETRIEVAL_CONTACT to your email.
CONTACT = os.environ.get("GRANT_RETRIEVAL_CONTACT", "")
UA = f"grant-retrieval-research/0.1 (contact: {CONTACT})"


def require_contact():
    if "@" not in CONTACT:
        sys.exit("Set GRANT_RETRIEVAL_CONTACT to a contact email; it is sent in the User-Agent header.")
FIELDS = ["foa_number", "status", "opportunity_id", "fetched_at"]


class Blocked(Exception):
    pass


def post(endpoint, payload):
    req = urllib.request.Request(f"{API}/{endpoint}", data=json.dumps(payload).encode(),
                                 headers={"User-Agent": UA, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            body = json.loads(r.read())
    except urllib.error.HTTPError as e:
        if e.code in (403, 429):
            raise Blocked(f"{e.code} from {endpoint}")
        raise
    finally:
        time.sleep(1.0)
    return body


def main(root, foa_list):
    require_contact()
    root = Path(root)
    out = root / "raw/grantsgov"
    out.mkdir(parents=True, exist_ok=True)
    log = root / "logs/grantsgov_fetch.csv"
    new = not log.exists()
    with open(log, "a", newline="") as fh:
        w = csv.DictWriter(fh, FIELDS)
        if new:
            w.writeheader()
        for foa in [l.strip() for l in open(foa_list) if l.strip()]:
            if (out / f"{foa}.json").exists():
                continue
            hits = post("search2", {"oppNum": foa, "rows": 25,
                                    "oppStatuses": "forecasted|posted|closed|archived"})
            hits = [h for h in hits.get("data", {}).get("oppHits", []) if h.get("number") == foa]
            status, oid = "not_found", ""
            if hits:
                oid = hits[0]["id"]
                body = post("fetchOpportunity", {"opportunityId": int(oid)})
                (out / f"{foa}.json").write_text(json.dumps(body))
                status = "ok"
            w.writerow({"foa_number": foa, "status": status, "opportunity_id": oid,
                        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
            fh.flush()
            print(foa, status, oid, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:3])
