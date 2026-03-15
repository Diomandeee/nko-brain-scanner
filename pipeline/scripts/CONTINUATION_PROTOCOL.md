# Continuation Protocol

1. Trigger
1.1 Must stop at a natural boundary or when the token limit is near.

2. Stop Output
2.1 Must include the CONTINUATION ANCHOR block with current phase, last completed checklist item, next active item, confidence, open uncertainties, and blocked by.

3. Continuation Command
3.1 Must resume exactly from the next active checklist item when the user says "continue".
3.2 Must re-anchor context by stating phase, last completed, next active, confidence, uncertainties, and blocked by.
3.3 Must not re-summarize completed work.

4. No Context Loss Rule
4.1 Must treat prior artifacts as authoritative unless explicitly revised.
4.2 Must ask for clarification if unsure about prior state.

5. Anchor Template
5.1
```
CONTINUATION ANCHOR
-------------------
Phase: [Current phase number and name]
Last Completed: [Most recent completed checklist item]
Next Active: [Next pending item]
Confidence: [Low / Medium / High]
Open Uncertainties: [List any unresolved ambiguities]
Blocked By: [Any blocking dependencies, or "None"]
```

6. Change History
6.1 2026-01-03 v0.1 Initial continuation protocol.
