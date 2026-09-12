# Changelog

## v2

Supersedes v1.

- **Retracted the v1 conclusion.** v1 reported a corrected harm gap of 0.26 from a
  single unblinded audit and read it as the grader concealing most of a real safety gap.
  Two further validations put the same quantity at −0.036, +0.274 and 0.035. The v1
  estimate is one of four, and the spread across them exceeds the effect.
- **Renamespaced `item_id`.** v1 identifiers collided across audits. 238 identifiers
  denoted different items in audits 2 and 3. Joining v1 files on item ID is invalid.
- **Added** audit-3 annotator sheets, the recovered framing key, the pre-registered
  codebook, the rescored grader labels, the uncensored 1024-token corpus, and per-budget
  grader labels at 200, 384, 512, 768 and 1024.
- **Added** a verification notebook that recomputes every headline statistic from the
  released CSVs.
- **Documented** two instrument changes that v1 left implicit: the reimplemented keyword
  scorer and the reformatted grader prompt.

## v1

Initial release. Conclusions superseded, see above.
