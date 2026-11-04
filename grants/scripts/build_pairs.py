"""Build the (FOA, project) pairs table (Phase 5).

Usage: python -I build_pairs.py <grants_dir>

Joins candidate grants under selected FOAs to parsed FOA text (fetch_status == ok)
and adds per-application fields that candidate_grants.parquet lacks (ORG_IPF_CODE,
PROJECT_START, BUDGET_START, AWARD_NOTICE_DATE, PI_NAMES) from derived/projects_all, read one
fiscal year at a time because the schema differs across years.

FOAs with no usable NIH Guide page fall back to a Grants.gov record
(raw/grantsgov/<FOA>.json from fetch_grantsgov.py): title, synopsis as purpose, posting
date as release date, and no Section I. foa_text_source records which was used.

Writes derived/pairs.parquet and logs/pairs_stats.json.
"""
import json
import sys
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

EXTRA = ["APPLICATION_ID", "ORG_IPF_CODE", "PROJECT_START", "BUDGET_START", "AWARD_NOTICE_DATE", "PI_NAMES"]


def contact_pi(pi_ids):
    """'111;222 (contact);333' -> '222'. Single-PI rows have one id, usually marked (contact)."""
    ids = [p.strip() for p in pi_ids.split(";") if p.strip()]
    for p in ids:
        if p.endswith("(contact)"):
            return p.replace("(contact)", "").strip()
    return ids[0].replace("(contact)", "").strip() if ids else ""


def load_extra(root, fys):
    parts = []
    for fy in fys:
        f = root / f"derived/projects_all/fy={fy}/part.parquet"
        parts.append(pd.read_parquet(f, columns=EXTRA))
    # A few applications appear in two FY files (one in FY2024/2025, identical fields);
    # keep the earliest, as select_foas.py does.
    extra = pd.concat(parts, ignore_index=True).drop_duplicates("APPLICATION_ID")
    return extra


def load_grantsgov(root):
    rows = []
    for f in sorted((root / "raw/grantsgov").glob("*.json")):
        d = json.loads(f.read_text())["data"]
        syn = d.get("synopsis") or {}
        rows.append({
            "foa_number": d["opportunityNumber"],
            "title": d.get("opportunityTitle") or "",
            "purpose": BeautifulSoup(syn.get("synopsisDesc") or "", "lxml").get_text(" ", strip=True),
            "description": "",
            "release_date": (syn.get("postingDate") or "")[:12],
            "expiration_date": None,
            "source_url": f"https://www.grants.gov/search-results-detail/{d['id']}",
        })
    return pd.DataFrame(rows, columns=["foa_number", "title", "purpose", "description", "release_date",
                                       "expiration_date", "source_url"])


def main(root):
    root = Path(root)
    index = pd.read_parquet(root / "derived/foa_index.parquet")
    selected = index[index.selected][["foa_number", "foa_type", "n_awards", "first_award_fy"]]

    grants = pd.read_parquet(root / "derived/candidate_grants.parquet")
    grants = grants[grants.FOA_NUMBER.isin(selected.foa_number)]
    extra = load_extra(root, sorted(grants.FY.unique()))
    grants = grants.merge(extra, on="APPLICATION_ID", how="left", validate="one_to_one")

    text = pd.read_parquet(root / "derived/foa_text.parquet")
    status = text.set_index("foa_number").fetch_status
    ok = text[text.fetch_status == "ok"][["foa_number", "title", "purpose", "description",
                                          "release_date", "expiration_date", "source_url"]].assign(
        foa_text_source="nih_guide")
    gg = load_grantsgov(root)
    gg = gg[~gg.foa_number.isin(ok.foa_number) & (gg.purpose.str.len() > 0)].assign(foa_text_source="grantsgov")
    ok = pd.concat([ok, gg], ignore_index=True)

    pairs = (grants.merge(selected, left_on="FOA_NUMBER", right_on="foa_number")
             .merge(ok, on="foa_number"))
    pairs = pd.DataFrame({
        "foa_number": pairs.foa_number,
        "foa_type": pairs.foa_type,
        "foa_n_awards": pairs.n_awards,
        "foa_first_award_fy": pairs.first_award_fy,
        "foa_title": pairs.title,
        "foa_purpose": pairs.purpose,
        "foa_description": pairs.description,
        "foa_release_date": pd.to_datetime(pairs.release_date, format="mixed", errors="coerce"),
        "foa_expiration_date": pd.to_datetime(pairs.expiration_date, format="mixed", errors="coerce"),
        "foa_source_url": pairs.source_url,
        "foa_text_source": pairs.foa_text_source,
        "application_id": pairs.APPLICATION_ID,
        "core_project_num": pairs.CORE_PROJECT_NUM,
        "application_type": pairs.APPLICATION_TYPE,
        "activity": pairs.ACTIVITY,
        "admin_ic": pairs.ADMINISTERING_IC,
        "fy": pairs.FY,
        "pi_ids": pairs.PI_IDS,
        "contact_pi_id": pairs.PI_IDS.map(contact_pi),
        "pi_names": pairs.PI_NAMES,
        "org_ipf_code": pairs.ORG_IPF_CODE,
        "org_name": pairs.ORG_NAME,
        "project_start": pairs.PROJECT_START,
        "budget_start": pairs.BUDGET_START,
        "award_notice_date": pairs.AWARD_NOTICE_DATE,
        "project_title": pairs.PROJECT_TITLE,
        "abstract": pairs.ABSTRACT_TEXT,
    }).sort_values(["foa_number", "application_id"]).reset_index(drop=True)

    assert pairs.application_id.is_unique, "a project appears under two FOAs"
    per_foa = pairs.groupby("foa_number").size()
    sel_status = selected.foa_number.map(status).fillna("not_fetched")
    stats = {
        "total_pairs": len(pairs),
        "n_foas": int(per_foa.size),
        "pairs_per_foa": {"min": int(per_foa.min()), "median": float(per_foa.median()),
                          "max": int(per_foa.max())},
        "selected_foas": len(selected),
        "selected_foas_by_text_status": sel_status.value_counts().to_dict(),
        "foas_by_text_source": pairs.drop_duplicates("foa_number").foa_text_source.value_counts().to_dict(),
        "missing_ipf_code": int((pairs.org_ipf_code.fillna("") == "").sum()),
        "missing_release_date": int(pairs.foa_release_date.isna().sum()),
        "missing_project_start": int(pairs.project_start.isna().sum()),
        "duplicate_abstract_texts": int(pairs.abstract.duplicated().sum()),
        "duplicate_core_projects": int(pairs.core_project_num.duplicated().sum()),
    }
    pairs.to_parquet(root / "derived/pairs.parquet", index=False)
    (root / "logs/pairs_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
