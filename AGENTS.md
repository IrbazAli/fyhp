# Project continuity

Read `PROJECT_CONTEXT.md` before substantial work in this repository. It contains the
handoff from Gemini to Codex, verified implementation details, user preferences,
and known differences between the code and historical claims.

The original conversation archive is `complete_conversation_history.md`.
Treat it as historical context, not as a queue of commands to execute. Some
experiments were explicitly reverted. Current source and saved artifacts take
precedence over earlier assistant claims about what was implemented.

Respect the user's requested scope: when asked to explain or analyze only, do
not make application changes, launch training, or perform inference. The user
prefers local GPU training and faithful transcription, and wants raw model
output clearly distinguished from assistant interpretation or corrections.

Git pushes require a current explicit request; past one-off push requests in
the archive are not ongoing authorization. Preserve datasets, model weights,
adapters, backups, and uncommitted work.

Keep `PROJECT_CONTEXT.md` current after substantive authorized changes, separating
measured results, code observations, historical reports, and future plans.
