# JOSS submission — status and remaining steps

Target: [Journal of Open Source Software](https://joss.theoj.org). JOSS reviews
the *software* — documentation, tests, installability, scientific effort — and
explicitly does not judge novelty. The 14-page technical paper is far too long
for JOSS; it becomes supporting documentation, and `paper.md` is the submission.

## Ready

| Requirement | Status |
| --- | --- |
| OSI licence in repository | Apache-2.0 (`LICENSE`) |
| Public repository | github.com/python35/IINTS-SDK |
| Substantial scholarly effort | 959 commits, all since 2026-02-01 |
| Automated tests | 176 test modules; mypy and flake8 clean |
| Documentation | `README.md`, `docs/` (80+ pages), API reference |
| Contribution guidelines | `CONTRIBUTING.md` |
| Code of conduct | `CODE_OF_CONDUCT.md` |
| Citation metadata | `CITATION.cff` (v1.5.37) |
| Paper | `paper.md`, 524 words — inside the 250–1000 range |
| Bibliography | `paper.bib`, 8 entries, every DOI resolved against doi.org |
| Archive metadata | `.zenodo.json` |

## Only you can do these

**1. ORCID.** `paper.md` and `CITATION.cff` both carry the placeholder
`0000-0000-0000-0000`. JOSS requires a real ORCID for the corresponding author.
Register free at <https://orcid.org/register>, then replace it in both files.

**2. Zenodo archive and DOI.** JOSS requires an archived release.

- Sign in to <https://zenodo.org> with GitHub, and enable the toggle for
  `python35/IINTS-SDK` under Settings → GitHub.
- Cut a GitHub release from the existing `v1.5.37` tag. Zenodo mints a DOI
  automatically and reads `.zenodo.json` for the metadata.
- Put the DOI into `CITATION.cff` (uncomment the `doi:` line) and into the JOSS
  submission form.

**3. Submit.** <https://joss.theoj.org/papers/new> — repository URL, the
Zenodo DOI, and the release version. Review happens in the open on GitHub.

## Worth doing before you submit

**Evidence of use beyond the author.** JOSS reviewers look for it. The desktop
beta downloads, any classroom or supervisor use, or an issue opened by someone
else all count. If none exists yet, a short "Who uses this" section in the
README describing the intended users is better than silence.

**Public issue history.** A repository with no issues reads as a single-author
dump. Opening issues for the known open problems is a positive signal, not a
negative one — start with the two in `docs/EXPERIMENTAL_RESULTS.md`: the
hypoglycaemia-detection result and the supervisor severe-hypoglycaemia rate.

**Fix the two documentation defects.** `docs/PAPER_ERRATA.md` records two places
where the technical paper describes code that does not exist: the IOB/COB loss
terms in equations (56)–(58), and the three-compartment Hovorka meal chain in
Table 5. A reviewer who checks the paper against the source will find both. The
paper's LaTeX source is not in this repository, so they have to be fixed
wherever that source lives.

## If you later write a results paper

JOSS papers do not contain results, so nothing below affects this submission.
But for any venue that does: `docs/EXPERIMENTAL_RESULTS.md` is the honest
summary, and the two unfavourable findings there — the forecaster losing to
persistence on hypoglycaemia detection, and the supervisor arms' severe
hypoglycaemia rate — have to lead, not trail. Reporting them is the strongest
credibility argument available, and burying them is the fastest way to lose a
referee.
