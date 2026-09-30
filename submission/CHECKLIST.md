# Submission checklist: Sustainable Operations and Computers (KeAi / Elsevier)

Checked against the Guide for Authors at
https://www.keaipublishing.com/en/journals/sustainable-operations-and-computers/guide-for-authors/
(retrieved 30 September 2026). Re-read the live guide on the day of submission, because journals edit it.

## Done in the package

| Guide item | Status | Where |
|---|---|---|
| LaTeX with Elsevier class `elsarticle` | done | `manuscript/carma_manuscript.tex` (numbered-reference style `elsarticle-num`) |
| Single-column text, simple layout | done | `preprint` option |
| Numbered sections and subsections, numbered cross-references (no "above"/"the text") | done | sections 1-7, appendices A-D; cross-references use `\ref` |
| Introduction states objectives and background, no detailed survey or results summary | done | Section 1 (related work is Section 2) |
| Methods reproducible; published methods cited | done | Section 3, code and data in the repository |
| Theory, Results, Discussion, Conclusions | done | Sections 4-7; Conclusions is short |
| Appendices lettered; tables and figures numbered A.1, B.1 | done | Appendices A-D |
| Title concise, no abbreviations or formulae | done | |
| Author, affiliation, corresponding author, e-mail, ORCID | done | postal address still to add (see below) |
| Highlights: separate editable file, name contains "Highlights", 3-5 bullets of at most 85 characters | done | `Highlights.docx` (5 bullets, longest 81 characters) |
| Abstract concise, factual, no references, abbreviations defined | done | 243 words; CARMA defined in the abstract |
| Graphical abstract (optional): at least 531 x 1328 px, readable at 5 x 13 cm | done | `Graphical_abstract.{pdf,png,tiff}` (1535 x 590 px) |
| At most 6 keywords, American spelling, no "and"/"of" | done | 6 keywords |
| Non-standard abbreviations defined in a footnote on the first page | done | title footnote |
| Acknowledgements in a separate section before the references | not needed | none to acknowledge |
| Funding statement in the recommended wording | done | "did not receive any specific grant..." |
| Declaration of competing interests, plus the Word declaration file | done | section in the manuscript and `Declaration_of_interest.docx` |
| CRediT author contributions | done | |
| Data availability | done | public sources cited as `[dataset]` references; code and results in the repository |
| One language variant, no mixture (American or British) | done | American throughout, including figures and tables |
| SI units | done | |
| Footnotes sparing | done | one footnote (abbreviations) |
| Tables: editable text, no vertical rules or shading, notes below the table | done | `booktabs` |
| Figures: separate files, vector PDF, fonts embedded (TrueType/Type 1), Times-like lettering | done | `figures/Fig_*.pdf` |
| Figure captions supplied separately, brief title plus description | done | `Figure_captions.txt` (captions are also in the source) |
| Figures accessible with impaired color vision | done | Okabe-Ito palette checked with a validator, plus hatching, markers and line styles |
| References numbered in square brackets; every citation in the list and vice versa | done | compiled with no undefined citations |
| Reference metadata correct, DOIs | done | all 58 DOIs in the bibliography checked against Crossref (title, year, first author) |
| Data references marked `[dataset]`, web references with last-accessed date, preprints marked | done | 8 dataset and web references; no preprint-only references are cited |
| Submission declaration (not published, not under consideration elsewhere) | in the cover letter | `Cover_letter.docx` |

## Open items for you

1. **Postal address.** The guide requires the full postal address of each affiliation. Add the street
   and postcode of British University Vietnam in `carma_manuscript.tex` (`\affiliation{...}`, marked with a TODO comment).
2. **Generative AI declaration.** The journal encourages a statement, placed before the references, if generative AI was
   used in the writing process; no statement is required if nothing needs to be disclosed (basic grammar, spelling and
   reference checks are exempt). No statement was added. Whether to add one, and what it says, is your decision and
   your responsibility.
3. **Open Week and APC.** The guide gives the open access fee as usually USD 700, excluding taxes. I could not find any
   mention of an Open Week or a fee waiver on the pages of the journal that I can access. Confirm the terms, the exact
   dates and the eligibility conditions on KeAi's own announcement before relying on it, and check whether the
   submission date or the acceptance date is the one that counts.
4. **License.** After acceptance you choose CC BY or CC BY-NC-ND.
5. **Preprint.** The journal allows free preprint posting on SSRN at submission. The public GitHub repository contains
   the manuscript source; decide whether you want that to stay public before submitting.
6. **Tables.** The guide asks authors to be sparing with tables and not to duplicate results given in the text.
   The paper has 11 tables (3 in appendices). Consider moving the related-work table or the fleets table if a reviewer
   finds it heavy.
7. **Editorial Manager.** Upload `Manuscript.pdf` or the LaTeX source in `manuscript/`, the figure files from `figures/`,
   `Highlights.docx`, `Declaration_of_interest.docx`, the graphical abstract and the cover letter. Line numbers are on.
