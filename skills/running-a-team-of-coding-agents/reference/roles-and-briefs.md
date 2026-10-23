# Roles, briefs, and reviews

## The role split

| Role | Who | Does | Never does |
|---|---|---|---|
| Builder | A fast coding agent (Codex, or a cheaper Claude model) | Writes code and tests from a brief, applies review fixes | Commit, push, stash, deploy, or work outside its worktree |
| Lead | Claude Code in the person's session | Plans, writes briefs, reads every diff, commits, fixes delicate code (money paths, auth, migrations) itself | Deploy, or decide what belongs to the person |
| Reviewer | A strong model with a fresh context | Reviews each task and the whole branch before a pull request | Edit code |
| Human | The accountable person | Money, direction, production risk, deploys, user-visible changes | Routine technical choices |

One agent can play builder and lead when only one is available. Keep the reviewer separate, with a fresh context that has not seen the builder's reasoning, so the review is not anchored on it.

## Brief template

A brief is everything the builder will know. Keep it short and point at files rather than retyping them; every extra page is reread on every turn.

```markdown
# Task: <one line>

Worktree: <path>  Branch: <name>
Read first: AGENTS.md, <spec path>, <plan path> (your task is section N).

Build: <what, in two to five sentences, with the files you expect to change>.
Tests: write the failing test first in <file>.test.<ext>, see it fail, then implement.
Do not touch: <files, directories, behaviour that must not change>.
Run only: your own test files, the type checker, and the linter. The script runs the full suites.
If the plan is wrong about a name or file, follow the real code, keep the intent, and say what differed.
Before changing a rule, list every place that answers the same question (file:line) and change the owner.

Report in plain words: files changed, test counts you saw, anything that differed from the plan, anything unsure.
```

## Review prompt

Ask for the verdict first, in a fixed form a script can parse, then findings ranked by severity. Give the reviewer the diff and the spec, not the whole repository.

```markdown
Review this diff against the spec at <path>. Read the diff (git diff <base>...<branch>) and open other files
only to check a specific claim.

Check, in order: correctness (what input gives a wrong result or crash), one owner per decision (does this
change decide something already decided elsewhere?), maintainability (file size, duplication, comments that
explain why), and any text a user will see.

First line of your answer, exactly one of:
VERDICT: PASS
VERDICT: FIX
Then each finding as: file:line, what goes wrong, the concrete input that triggers it, and the fix.
```

The script treats a missing or malformed verdict line as a failure. A reviewer that timed out or crashed has not approved anything.

## Fix rounds

A fix round gives the builder only the review's findings and the current diff. After the round, review the new diff again. After two failed rounds on the same rule, stop: the rule is answered in more than one place, and the next round will patch a third. Find every place that answers the question, move the answer into one owning module as its own small change, and then resume.

## Cost habits

Measured on one project, about 95% of model spend paid for agents rereading context, so cost follows context size more than task size.

- Narrow briefs that point at files instead of pasting them.
- Reviews that read the diff, not the whole codebase.
- The cheaper model or tool builds; the expensive one reviews.
- Waiting (CI, merges, long test suites) happens in shell scripts, not in agent sessions that reread their history on every check.
- One review per task and one per branch. Don't rerun a stage that a later stage repeats.
