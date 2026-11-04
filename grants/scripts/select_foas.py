"""Choose FOAs to fetch text for (Phase 2) and report pair-count stats.

Usage: python -I select_foas.py <grants_dir> [first_fy] [last_fy]

Candidate grants: NIH institutes only, research mechanisms, new or renewal
applications (type 1/2), not subprojects, with an FOA number and an abstract.
An FOA qualifies with 5-200 such grants and an RFA/PAR/PA/PAS number; PAs with
more than 200 are flagged as parent announcements.

Outputs:
  derived/candidate_grants.parquet  the candidate grant rows (pairs come from these)
  derived/foa_index.parquet         every FOA seen, with award count and excluded_reason
  logs/phase2_stats.json
"""
import json
import sys
from pathlib import Path

import pandas as pd

NON_NIH_ICS = {"VA", "FD", "HS", "OH", "CE", "DP", "IP", "CK", "DD", "GH", "PS", "TS", "EH", "SH", "TP", "JG"}
RESEARCH_MECHANISMS = {"NON-SBIR/STTR RPGS", "SBIR-STTR RPGS", "RESEARCH CENTERS", "OTHER RESEARCH-RELATED"}
KEPT_TYPES = {"RFA", "PAR", "PA", "PAS"}
MIN_AWARDS, MAX_AWARDS = 5, 200
CAPS = [10, 20, 40]
COLS = ["APPLICATION_ID", "CORE_PROJECT_NUM", "FY", "FOA_NUMBER", "APPLICATION_TYPE", "SUBPROJECT_ID",
        "ADMINISTERING_IC", "FUNDING_MECHANISM", "ACTIVITY", "PI_IDS", "ORG_NAME", "PROJECT_TITLE", "ABSTRACT_TEXT"]


def main():
    grants = Path(sys.argv[1])
    first, last = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (2008, 2025)
    parts = [pd.read_parquet(grants / "derived" / "projects_all" / f"fy={y}" / "part.parquet", columns=COLS)
             for y in range(first, last + 1)]
    d = pd.concat(parts, ignore_index=True)
    d["FOA_NUMBER"] = d["FOA_NUMBER"].str.upper()
    d = d[(d["FOA_NUMBER"] != "") & (d["ABSTRACT_TEXT"] != "") & (d["SUBPROJECT_ID"] == "")
          & d["APPLICATION_TYPE"].isin(["1", "2"])
          & ~d["ADMINISTERING_IC"].isin(NON_NIH_ICS)
          & d["FUNDING_MECHANISM"].isin(RESEARCH_MECHANISMS)]
    # One row per application; the same application can repeat across FY files.
    d = d.sort_values("FY").drop_duplicates("APPLICATION_ID")
    d["FOA_TYPE"] = d["FOA_NUMBER"].str.extract(r"^([A-Z]+)-", expand=False).fillna("OTHER")
    d.to_parquet(grants / "derived" / "candidate_grants.parquet", index=False)

    idx = (d.groupby("FOA_NUMBER")
           .agg(foa_type=("FOA_TYPE", "first"), n_awards=("APPLICATION_ID", "size"),
                first_award_fy=("FY", "min"), last_award_fy=("FY", "max"),
                admin_ic=("ADMINISTERING_IC", lambda s: s.mode().iat[0]))
           .reset_index().rename(columns={"FOA_NUMBER": "foa_number"}))
    reason = pd.Series("", index=idx.index)
    reason[~idx["foa_type"].isin(KEPT_TYPES)] = "type_not_kept"
    reason[(reason == "") & (idx["n_awards"] > MAX_AWARDS) & idx["foa_type"].eq("PA")] = "parent_pa"
    reason[(reason == "") & (idx["n_awards"] > MAX_AWARDS)] = "too_many_awards"
    reason[(reason == "") & (idx["n_awards"] < MIN_AWARDS)] = "too_few_awards"
    idx["excluded_reason"] = reason
    idx["selected"] = reason == ""
    idx.to_parquet(grants / "derived" / "foa_index.parquet", index=False)

    sel = idx[idx["selected"]]
    stats = {
        "fy_range": [first, last],
        "candidate_grants": len(d),
        "foas_seen": len(idx),
        "excluded": idx["excluded_reason"].value_counts().to_dict(),
        "selected_foas": len(sel),
        "selected_grants": int(sel["n_awards"].sum()),
        "selected_by_type": sel.groupby("foa_type").agg(foas=("foa_number", "size"), grants=("n_awards", "sum")).to_dict("index"),
        "selected_by_first_award_fy": sel.groupby("first_award_fy").agg(foas=("foa_number", "size"), grants=("n_awards", "sum")).to_dict("index"),
        "pairs_under_cap": {cap: int(sel["n_awards"].clip(upper=cap).sum()) for cap in CAPS},
        "pairs_uncapped": int(sel["n_awards"].sum()),
        "awards_per_foa_percentiles": sel["n_awards"].quantile([0, .25, .5, .75, .9, 1]).to_dict(),
    }
    (grants / "logs" / "phase2_stats.json").write_text(json.dumps(stats, indent=1, default=int))
    print(json.dumps(stats, indent=1, default=int))


if __name__ == "__main__":
    main()
