# Codebook hashes (SHA-256)

The codebook is the written set of label definitions the annotators followed.
It was fixed before labelling began; these hashes let anyone check that the
files here are the ones that were used.

| File | Used by | SHA-256 |
|---|---|---|
| CODEBOOK_v1_as_used_by_A1.md | A1 (120 items) | `846c3361dac800731e1f36336ad7c5694489f02666eb56b71e78953d430e39dc` |
| CODEBOOK_v1_with_helper_column_A3.md | A3 (70 items) | `14b96ddbcc0be9980c0f162af221b87751aa7670cca0219980829c2105636854` |

The second file adds one line describing an optional helper column
(`final_number_found`); the label definitions are identical.

Check with: `sha256sum CODEBOOK_*.md`
