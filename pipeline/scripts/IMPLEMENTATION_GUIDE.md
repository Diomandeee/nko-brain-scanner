# Implementation Guide

1. Implementation Rules
1.1 Decision Signals
1.1.1 Must treat three or more independent agreeing signals as actionable commitment.
1.1.2 Should treat two agreeing signals as provisional commitment.
1.1.3 Must treat a single signal as observation only.
1.1.4 Must pause when signals conflict.
1.2 Conflict Resolution
1.2.1 Must choose the lowest-commitment reversible option when signals conflict.
1.2.2 Must document the conflict and chosen action in checklist notes.
1.3 Done Definition
1.3.1 Must have artifact created and accessible.
1.3.2 Must have validation condition satisfied and documented.
1.3.3 Must mark the checklist item as Complete.
1.4 Blocked Definition
1.4.1 Must mark an item Blocked when required inputs or dependencies are missing.
1.4.2 Must list the blocking dependency and the expected unblock condition.

2. Decomposition Rules
2.1 Atomic Unit Rule
2.1.1 Must define an atomic unit as a single, verifiable output that can be validated independently and rolled back without affecting siblings.
2.2 Scope Limit
2.2.1 Must keep each checklist item to one atomic unit.
2.2.2 Should split work if a checklist item requires multiple outputs.
2.3 Required Depth
2.3.1 Must decompose to Phase -> Step -> Substep -> Artifact before implementation begins.

3. Validation Rules
3.1 Phase Zero Documents
3.1.1 Must include required sections and change history.
3.1.2 Validation: Manual checklist verification and file existence.
3.2 Code Changes
3.2.1 Should run the smallest relevant script or test command available.
3.2.2 Must document when validation is skipped and why.
3.3 Data Outputs
3.3.1 Should validate presence of required schema keys and counts.
3.3.2 May add lightweight validation scripts if none exist, then remove after use.

4. Traceability Rules
4.1 Must link each decision to a requirement, invariant, or locked guideline.
4.2 Must record trace links in the artifact or checklist notes.

5. Change History
5.1 2026-01-03 v0.1 Initial implementation guide derived from constitution.
