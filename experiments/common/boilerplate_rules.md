# Boilerplate and cleaning rules (Plan 1 step 1)

Implemented in `experiments/common/text.py`; texts are built by `build_text.py`. Every plan must build query and document text through this module.

## Abstracts (document side)

How the rules were found: normalised sentences counted over the 81,333 abstracts under qualifying FOAs, one count per abstract. No content sentence appears in more than 2% of abstracts. The only frequent items are section headers and export artifacts:

| Item | Abstracts | Rule |
|---|---:|---|
| `DESCRIPTION (provided by applicant):` (and `[unreadable]`-prefixed form) | 40% | Removed when leading. Era-specific (mostly pre-2015), so it could leak the award year. |
| `Public Health Relevance:` trailer | 17–19% | Cut from the heading to the end. Starts at about 80% of the text in more than 99% of cases. |
| `Project Summary`, `Project Summary/Abstract`, `Abstract`, `Summary`, `Project Abstract`, `Description:`, `Overall` | 1–7% each | Removed when leading (repeated headers too). |
| `[unreadable]` | 3.5%+ | Removed everywhere (ExPORTER export artifact). |
| `Relevance (See instructions):`, `Project Narrative`, `Narrative:` trailers | 0.3–0.8% each | Cut from the heading to the end, like PHR. |
| `(End of Abstract)` | 0.5% | Removed. |

A trailer is cut only if what remains is at least 40% of the text. Document text is `project title + "\n" + cleaned abstract`, truncated to 767 Qwen3 tokens (768 with the appended `<|endoftext|>`).

**Not masked:** institution names (see `plans/limitations.md`).

## FOAs (query side)

Fields used: `title`, `purpose`, and for variant B `description` (Section I). `boilerplate` (Sections II–VIII: award information, eligibility, how to apply, review criteria) is never used.

1. **Parser artifacts.** The NIH Guide HTML carries hidden template comments that the parser kept as text: "This variable defines that we need to start a new row." (69% of purposes), "Closing the div tags for a table element, if a vertical element starts after table element." (44%), "If there are no labels defined for table element, we treat that as 1 row and append the text.", stray `</div>` and `End Row`. Removed everywhere, titles included.
2. **Frequent sentences.** Normalised sentences (lowercase, whitespace collapsed, at least 20 characters) found in ≥ 2% of FOAs (counted once per FOA over purpose + Section I) are removed. The list, with counts, is in `foa_boilerplate_sentences.json`. Examples: "Other Information for Award Authorities and Regulations.", the clinical-trial policy sentences, diversity / PEDP paragraphs, generic K-award and R25 program descriptions, NIDA council guidelines, section headings ("Specific Areas of Research Interest", "Non-responsive applications"), and bare institute heading lines ("National Institute of Mental Health (NIMH)"); the institute-specific paragraphs under those headings stay. The counts use only unlabeled FOA text from all FOAs with text (FY2008–2024), never award labels.
3. **Query variants.** A = title + purpose. B = title + purpose + Section I. Both are truncated to 490 Qwen3 tokens, so prompt (21) + text + `<|endoftext|>` ≤ 512. Truncation happens before any model sees the text, so BM25 and dense models get identical text.

## BM25 tokenisation

Lowercase; tokens `\b[a-z0-9][a-z0-9-]+\b` (length ≥ 2, hyphens kept); sklearn English stop words removed; no stemming. Okapi BM25 with k1 = 1.2, b = 0.75.
