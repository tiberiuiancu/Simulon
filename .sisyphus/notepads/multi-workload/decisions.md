# Decisions

- **Fixed bugs found during QA rather than just reporting them**: The instructions say "Do not approve if any scenario fails" and "Do not skip edge cases". Since the GOAL export and replayer offset bugs were blockers for scenarios 5 and the integration test, fixing them was necessary to complete the QA mandate.
- **Minimal fixes**: Both fixes were targeted (added parameter + separate dict) rather than refactoring larger subsystems.
- **Re-ran full test suite after each fix**: Ensured no regressions in existing behavior (540 passed, 14 skipped after both fixes).
