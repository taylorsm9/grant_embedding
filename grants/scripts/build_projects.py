"""Parse ExPORTER zips into parquet (Phase 1, steps 3-5).

Usage: python -I build_projects.py <grants_dir>

Outputs (all regenerated from raw/):
  derived/projects_all/fy=YYYY/part.parquet  every fiscal-year row, abstract joined
  derived/projects.parquet                   one row per CORE_PROJECT_NUM: the earliest
                                             FY row that has an abstract
  logs/phase1_stats.json                     validation counts for the README
"""
import json
import logging
import re
import sys
import zipfile
from pathlib import Path

import pandas as pd

# Columns renamed across years; everything else keeps its source name.
RENAME = {
    "OPPORTUNITY NUMBER": "FOA_NUMBER",
    "CFDA_CODE": "ASSISTANCE_LISTING_NUMBER",
    "PI_NAMEs": "PI_NAMES",
    "FUNDING_ICs": "FUNDING_ICS",
}
NUMERIC = ["TOTAL_COST", "TOTAL_COST_SUB_PROJECT", "DIRECT_COST_AMT", "INDIRECT_COST_AMT", "SUPPORT_YEAR"]
DATES = ["PROJECT_START", "PROJECT_END", "BUDGET_START", "BUDGET_END", "AWARD_NOTICE_DATE"]
# e.g. 1R01CA123456-01A1 -> R01CA123456; 5P01AG012345-03S1 -> P01AG012345
CORE_RE = re.compile(r"^\d?([A-Z]\d{2}[A-Z]{2}\d{6})")


def read_zip_csv(path):
    with zipfile.ZipFile(path) as z:
        name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
        raw = z.read(name)
    try:
        text = raw.decode("utf-8-sig")
        enc = "utf-8"
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
        enc = "latin-1"
    from io import StringIO
    df = pd.read_csv(StringIO(text), dtype=str, keep_default_na=False, on_bad_lines="warn")
    df.columns = [c.strip() for c in df.columns]
    return df, enc


def load_year(root, fy, known):
    proj, enc_p = read_zip_csv(root / "projects" / f"RePORTER_PRJ_C_FY{fy}.zip")
    proj = proj.rename(columns=RENAME)
    unmapped = sorted(set(proj.columns) - known)
    if unmapped:
        logging.warning("FY%d unmapped columns: %s", fy, unmapped)

    abs_path = root / "abstracts" / f"RePORTER_PRJABS_C_FY{fy}.zip"
    if abs_path.exists():
        abstracts, enc_a = read_zip_csv(abs_path)
        abstracts["ABSTRACT_TEXT"] = abstracts["ABSTRACT_TEXT"].str.strip()
        abstracts = (abstracts[abstracts["ABSTRACT_TEXT"] != ""]
                     .drop_duplicates("APPLICATION_ID"))
        proj = proj.merge(abstracts, on="APPLICATION_ID", how="left")
    else:
        enc_a = None
        proj["ABSTRACT_TEXT"] = ""
    proj["ABSTRACT_TEXT"] = proj["ABSTRACT_TEXT"].fillna("")

    for c in proj.columns:
        if proj[c].dtype == object:
            proj[c] = proj[c].str.strip()
    derived = proj["FULL_PROJECT_NUM"].str.extract(CORE_RE, expand=False).fillna("")
    proj["CORE_PROJECT_NUM"] = proj["CORE_PROJECT_NUM"].where(proj["CORE_PROJECT_NUM"] != "", derived)
    proj["FY"] = pd.to_numeric(proj["FY"], errors="coerce").fillna(fy).astype("int16")
    for c in NUMERIC:
        if c in proj:
            proj[c] = pd.to_numeric(proj[c], errors="coerce")
    for c in DATES:
        if c in proj:
            proj[c] = pd.to_datetime(proj[c], errors="coerce")
    proj["SOURCE_FY_FILE"] = fy
    return proj, {"projects_encoding": enc_p, "abstracts_encoding": enc_a, "unmapped_columns": unmapped}


def main():
    grants = Path(sys.argv[1])
    raw = grants / "raw" / "nih_exporter"
    out_all = grants / "derived" / "projects_all"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(grants / "logs" / "build_projects.log"), logging.StreamHandler()],
    )
    known = {
        "ACTIVITY", "ADMINISTERING_IC", "APPLICATION_ID", "APPLICATION_TYPE", "ARRA_FUNDED",
        "ASSISTANCE_LISTING_NUMBER", "AWARD_NOTICE_DATE", "BUDGET_END", "BUDGET_START",
        "CORE_PROJECT_NUM", "DIRECT_COST_AMT", "ED_INST_TYPE", "FOA_NUMBER", "FULL_PROJECT_NUM",
        "FUNDING_ICS", "FUNDING_MECHANISM", "FY", "IC_NAME", "INDIRECT_COST_AMT", "NIH_SPENDING_CATS",
        "ORG_CITY", "ORG_COUNTRY", "ORG_DEPT", "ORG_DISTRICT", "ORG_DUNS", "ORG_FIPS", "ORG_IPF_CODE",
        "ORG_NAME", "ORG_STATE", "ORG_ZIPCODE", "PHR", "PI_IDS", "PI_NAMES", "PROGRAM_OFFICER_NAME",
        "PROJECT_END", "PROJECT_START", "PROJECT_TERMS", "PROJECT_TITLE", "SERIAL_NUMBER",
        "STUDY_SECTION", "STUDY_SECTION_NAME", "SUBPROJECT_ID", "SUFFIX", "SUPPORT_YEAR",
        "TOTAL_COST", "TOTAL_COST_SUB_PROJECT",
    }
    years = sorted(int(p.stem[-4:]) for p in (raw / "projects").glob("RePORTER_PRJ_C_FY*.zip"))

    stats = {"per_year": {}, "top_foa": {}}
    keep = ["APPLICATION_ID", "CORE_PROJECT_NUM", "FY", "FOA_NUMBER", "SUBPROJECT_ID",
            "APPLICATION_TYPE", "PI_IDS", "ORG_NAME", "ABSTRACT_TEXT"]
    slim = []
    for fy in years:
        df, info = load_year(raw, fy, known)
        part = out_all / f"fy={fy}"
        part.mkdir(parents=True, exist_ok=True)
        df.to_parquet(part / "part.parquet", index=False)
        has_abs = df["ABSTRACT_TEXT"] != ""
        has_foa = df["FOA_NUMBER"] != ""
        stats["per_year"][fy] = {
            "rows": len(df),
            "pct_abstract": round(100 * has_abs.mean(), 1),
            "pct_foa": round(100 * has_foa.mean(), 1),
            "pct_subproject": round(100 * (df["SUBPROJECT_ID"] != "").mean(), 1),
            **info,
        }
        logging.info("FY%d rows=%d abstract=%.1f%% foa=%.1f%%", fy, len(df),
                     100 * has_abs.mean(), 100 * has_foa.mean())
        slim.append(df[keep].assign(HAS_ABSTRACT=has_abs).drop(columns="ABSTRACT_TEXT"))

    allrows = pd.concat(slim, ignore_index=True)
    stats["total_rows"] = len(allrows)
    stats["distinct_core_projects"] = int(allrows["CORE_PROJECT_NUM"].nunique())
    pis = allrows["PI_IDS"].str.split(";").explode().str.replace(r"\(contact\)", "", regex=True).str.strip()
    stats["distinct_pi_ids"] = int(pis[pis != ""].nunique())
    stats["distinct_orgs"] = int(allrows.loc[allrows["ORG_NAME"] != "", "ORG_NAME"].nunique())
    foa = allrows[allrows["FOA_NUMBER"] != ""]
    stats["top_foa"] = foa.groupby("FOA_NUMBER")["CORE_PROJECT_NUM"].nunique().nlargest(20).to_dict()

    # Dedup: earliest FY row with an abstract per core project; ties go to the
    # lowest APPLICATION_ID. Projects that never have an abstract are dropped here
    # but remain in projects_all.
    cand = allrows[allrows["HAS_ABSTRACT"] & (allrows["CORE_PROJECT_NUM"] != "")]
    cand = cand.assign(_aid=pd.to_numeric(cand["APPLICATION_ID"], errors="coerce"))
    chosen = cand.sort_values(["FY", "_aid"]).drop_duplicates("CORE_PROJECT_NUM")
    picks = set(zip(chosen["FY"], chosen["APPLICATION_ID"]))
    parts = []
    for fy in years:
        df = pd.read_parquet(out_all / f"fy={fy}" / "part.parquet")
        parts.append(df[[k in picks for k in zip(df["FY"], df["APPLICATION_ID"])]])
    dedup = pd.concat(parts, ignore_index=True)
    dedup.to_parquet(grants / "derived" / "projects.parquet", index=False)
    per_fy_after = dedup.groupby("FY").size().to_dict()
    for fy, n in per_fy_after.items():
        stats["per_year"].setdefault(int(fy), {})["rows_after_dedup"] = int(n)
    stats["dedup_rows"] = len(dedup)
    stats["dedup_pct_foa"] = round(100 * (dedup["FOA_NUMBER"] != "").mean(), 1)

    (grants / "logs" / "phase1_stats.json").write_text(json.dumps(stats, indent=1, default=str))
    logging.info("done: %d rows total, %d deduped projects", len(allrows), len(dedup))


if __name__ == "__main__":
    main()
