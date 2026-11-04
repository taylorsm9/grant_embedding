"""Extract clean text from saved NIH Guide pages (Phase 3, steps 3-4).

Usage: python -I parse_foas.py <grants_dir>

Reads raw/nih_guide/*.html plus logs/foa_fetch.csv and writes
derived/foa_text.parquet. Sections:
  purpose      the short "Funding Opportunity Purpose" from the overview
  description  Section I (Funding Opportunity Description): the scientific text
  boilerplate  Sections II-VIII (award, eligibility, how to apply, review, ...)
  body         the full announcement text
"""
import csv
import re
import sys
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

SECTION_I = re.compile(r"Section\s+I\.\s+(?:Notice of\s+)?Funding\s+Opportunity\s+Description", re.I)
SECTION_II = re.compile(r"Section\s+II\.\s+Award\s+Information", re.I)
PURPOSE = re.compile(r"(?:Funding\s+Opportunity|FOA)\s+Purpose\s*\n", re.I)
# Overview labels that can follow the purpose paragraph.
AFTER_PURPOSE = re.compile(r"\n\s*(?:Key Dates|Posted Date|Funding Opportunity Goal|Assistance Listing|"
                           r"Number of Applications|Part 2\.)", re.I)
# Older pages (~2006-2010) put it in an executive summary: "Purpose. The NIEHS invites ..."
OLD_PURPOSE = re.compile(r"Executive Summary\s*\n\s*(?:Purpose\.?\s*)?", re.I)
DATE = r":?\s*\n?\s*([A-Z][a-z]+\.?\s+\d{1,2},\s+\d{4})"
BLOCK_TAGS = ["p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "td", "th", "table", "ul", "ol",
              "section", "article", "blockquote", "dt", "dd"]


def clean(text):
    text = re.sub(r"[ \t\r\f\v\xa0]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return re.sub(r"\n{2,}", "\n", text).strip()


def extract(html):
    soup = BeautifulSoup(html, "lxml")
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    title = re.sub(r"^(Expired|Archived|Withdrawn)\s+", "", title)
    title = re.sub(r"^[A-Z]+(?:-[A-Z]{2})?-\d{2}-\d{3}:\s*", "", title)
    for t in soup(["script", "style", "nav", "header", "footer", "noscript"]):
        t.decompose()
    main = soup.find("main") or soup.body or soup
    # Source HTML wraps lines mid-sentence; keep line breaks only at block elements.
    for s in main.find_all(string=True):
        s.replace_with(s.replace("\n", " "))
    for t in main.find_all(BLOCK_TAGS):
        t.insert_after("\n")
    for t in main.find_all("br"):
        t.replace_with("\n")
    text = clean(main.get_text(""))

    # The table of contents also lists Section I/II, so take the I..II span with the
    # most text between them.
    best = (0, 0)
    for m in SECTION_I.finditer(text):
        nxt = SECTION_II.search(text, m.end())
        if nxt and nxt.start() - m.end() > best[1] - best[0]:
            best = (m.end(), nxt.start())
    description = text[best[0]:best[1]].strip()
    boilerplate = text[best[1]:].strip() if best[1] else ""

    purpose = ""
    m = PURPOSE.search(text)
    if m:
        end = AFTER_PURPOSE.search(text, m.end())
        purpose = text[m.end(): end.start() if end else m.end() + 3000].strip()
    elif m := OLD_PURPOSE.search(text):
        purpose = text[m.end():].split("\n", 1)[0].strip()

    def date(label):
        m = re.search(label + DATE, text)
        return m.group(1) if m else ""

    return {
        "title": title, "purpose": purpose, "description": description,
        "boilerplate": boilerplate, "body": text,
        "release_date": date(r"(?:Release/Posted|Posted|Release) Date"),
        "expiration_date": date(r"Expiration Date"),
    }


def main():
    grants = Path(sys.argv[1])
    with open(grants / "logs" / "foa_fetch.csv", newline="") as f:
        log = {r["foa_number"]: r for r in csv.DictReader(f)}
    rows = []
    for foa, r in log.items():
        row = {"foa_number": foa, "source_url": r["source_url"], "fetch_status": r["status"]}
        path = grants / "raw" / "nih_guide" / f"{foa}.html"
        if r["status"] == "ok" and path.exists():
            row.update(extract(path.read_bytes()))
            if not row["description"]:
                row["fetch_status"] = "no_description"
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_parquet(grants / "derived" / "foa_text.parquet", index=False)
    print(df["fetch_status"].value_counts().to_dict())


if __name__ == "__main__":
    main()
