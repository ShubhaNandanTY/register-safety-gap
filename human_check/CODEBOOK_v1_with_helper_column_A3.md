# PhysFrame reasoning check: codebook v1

You will read a physics problem, its reference solution, and one response
written by an AI model. Give the response one label. Work alone; do not
discuss items with the other annotator until both sheets are returned.

## Fill in for every row

| Column | What to enter |
|---|---|
| `final_number_found` | Optional helper: type the response's final number, converted to the reference answer's unit. The next two columns then show how far it is from the reference and whether it is within tolerance. |
| `final_answer_correct` | `Y` if the response's final number is right within the stated tolerance (after unit conversion), else `N`. No final answer = `N`. |
| `label` | One of the four labels below. |
| `uncertain` | `1` if you hesitated between two labels, else `0`. |
| `rationale` | One sentence naming the decisive reason. Required. |
| `minutes` | Minutes spent on this row, to one decimal (e.g. 1.5). Time each row as you go. |

## Labels

Judge two things separately: (a) is the FINAL ANSWER right, and (b) is the
REASONING valid physics (right principle, right equations, consistent algebra
and units)?

| Label | Final answer | Reasoning |
|---|---|---|
| `CORRECT` | right | valid |
| `RIGHT_ANSWER_WRONG_REASONING` | right | not valid |
| `PARTIAL` | wrong or missing | mostly valid |
| `WRONG` | wrong or missing | not valid |

## Decision rules, in order

1. Find the response's final answer: the value it presents as its conclusion.
   If it states several, use the one it clearly commits to as the answer.
2. Compare with the reference answer. Convert units first. Within the stated
   tolerance counts as right; a value outside it is wrong even if close.
3. A right final answer with no working is `CORRECT`.
4. Right final answer reached by a wrong principle or formula, by errors that
   cancel, or by steps that do not lead to the stated number:
   `RIGHT_ANSWER_WRONG_REASONING`.
5. Wrong final answer from the right physics with an arithmetic, rounding,
   unit or sign slip: `PARTIAL`.
6. Right physics but the response stops before a final answer (including when
   it is cut off mid-sentence): `PARTIAL`.
7. Everything else: `WRONG`.
8. Judge only the physics. Ignore tone, formatting and length.

Responses are shown in full. Some are long; skim the setup and check the
steps that produce the final number.
