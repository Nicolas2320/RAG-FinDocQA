# Known limitations

## Table-flattening ambiguity in PDF text extraction (Day 4)

Some 10-K pages present financial data in multi-column tables where several
distinct metrics (e.g. "Property, Plant and Equipment - net" and "Capital
Spending") are laid out side by side, broken down further by year and region.

When these pages are extracted with PyMuPDF's `get_text()` (used in
`src/chunking/chunking_metadata.py`), the visual grid of the table collapses
into linear text. Column headers and their corresponding data rows can end up
out of their original left-to-right order, and multiple rows of numbers appear
consecutively without an explicit marker of which column each value belongs to.

### Concrete example

3M's FY2018 10-K, page index 38 ("Geographic Area Supplemental Information"),
contains this table:

| (Millions) | PP&E - net 2018 | PP&E - net 2017 | Capital Spending 2018 | Capital Spending 2017 | Capital Spending 2016 |
|---|---|---|---|---|---|
| Total Company | $8,738 | $8,866 | $1,577 | $1,373 | $1,420 |

Once flattened to plain text, the "Property, Plant and Equipment - net" header
appears before the "Capital Spending" header, even though visually Capital
Spending is the column closer to the metric asked about. When asked "What is
the FY2018 capital expenditure amount for 3M?", the generation step initially
answered **$8,738 million** (the PP&E - net figure) instead of the correct
**$1,577 million** (the Capital Spending figure) — not a hallucination, but a
misread of a genuinely ambiguous flattened table. The correct figure is also
present, unambiguously, in the primary Consolidated Statement of Cash Flows
(page index 59), but that page ranks lower in retrieval because its narrower,
literal phrasing ("Purchases of property, plant and equipment (PP&E)") scores
lower against the query than page 38's narrative language ("capital spending
was within the United States...").

### Root cause

Two compounding effects, isolated independently:

1. **Retrieval bias toward narrative text.** Both the bi-encoder (BAAI/bge-small-en-v1.5)
   and the cross-encoder reranker (BAAI/bge-reranker-base) score natural-language
   narrative passages higher than dense numeric/tabular fragments, regardless of
   chunk size or added context — confirmed by testing an isolated, noise-free
   snippet of the correct answer, which still scored lower (0.0004) than a full,
   noisy narrative page (0.96).
2. **Lossy table-to-text flattening.** `get_text()` extracts text in a reading
   order that does not always preserve the visual column alignment of tables,
   so the LLM receives a chunk where the correct numeric value cannot be
   reliably distinguished from an adjacent, similarly-labeled value without
   seeing the original grid.

### Status

Not fixed in this MVP — scoped out to keep the 1-week timeline. Both retrieval
(metadata filter + dense search + rerank) and generation (strict grounding +
citation prompting) work correctly for narrative-style evidence (validated end
to end on 3M, Best Buy, and MGM Resorts test questions); this failure mode is
specific to multi-column numeric tables.

### Future work

- Structure-aware table extraction (e.g. `pdfplumber`, `camelot`, or
  `unstructured`) to preserve row/column relationships instead of raw text flow.
- Multimodal extraction: render the page as an image and let a vision-capable
  LLM read the table directly, preserving its visual layout.
- A generation-time safeguard: instruct the LLM to flag low confidence when a
  retrieved chunk contains multiple numeric columns with similar labels, rather
  than silently picking one.
