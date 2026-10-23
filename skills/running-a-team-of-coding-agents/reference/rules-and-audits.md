# Rules, audits, and boundary tests

## Writing a rule after a failure

Agents start every session with no memory, so a lesson exists only once it is written where they read it: the repository's AGENTS.md or CLAUDE.md for repository rules, the brief template for build rules, and the review prompt for review rules. One incident is enough to justify a rule. Write it as what to do, with the reason in one clause, so an agent can apply it to a case the incident did not cover:

> A `"use server"` file exports only async functions, because any other export is registered as a server action and fails at runtime even though the type checker, linter, and build all pass.

When a rule can be checked mechanically, add the check (a test, a lint rule, or a CI step) as well, and keep the written rule for the judgement part.

## Seed rules for AGENTS.md

A starting set that most production repositories can adopt as written:

- Every agent works in its own git worktree and branch. Nobody shares a checkout, and nobody uses the shared stash.
- Every logic module has a colocated test file, and new code is fully covered. Components stay thin; behaviour lives in pure, tested functions.
- Each decision (which files make up a unit, which setting wins, who may see a record) is made in exactly one module. Everything else calls it. New cases extend the owner, not the callers.
- Stages hand each other plain data, not each other's internals.
- Risky new behaviour gets an off switch that restores the old behaviour exactly, with a test proving it.
- Settings are read once at the start of a job and passed down.
- Comments explain why, in plain words, and never cite process history ("fix round 2").
- No change a user can see merges without the accountable person's explicit approval of that change.
- Nobody but the accountable person deploys.

## Structure audits

Agents add a condition wherever they happen to be looking, so a decision that started in one module drifts into several. Schedule an audit as its own piece of work, and start one early when a feature fails review repeatedly.

1. List the decisions the code makes ("is this file a prompt?", "which price applies?").
2. For each, list every place that answers it, with file and line.
3. Any decision answered in more than one place gets one owning module. Merging the copies is its own small pull request, before any feature work on that decision continues.
4. Write the owner into AGENTS.md so the next agent calls it instead of re-deriving it.

## Boundary tests

Unit tests, type checks, lint, a database suite, a production build, and review can all pass while the system fails where its parts meet. Bugs of this kind found on one launch night:

| Boundary | What broke | Test that catches it |
|---|---|---|
| Code and data | A query depended on a row that only the seed script created | Run the feature against a database built by migrations alone |
| Build and runtime | A page was prerendered as static HTML because the build ran without runtime settings | Build with production settings and request the page from the built server |
| Tool and client | A tool returned structured output holding only an exit code | Call the tool through a real client and assert on the payload |

Also test new integrations against the real service in its test mode (payments, auth, email). A single test-mode call can find bugs that every mocked test missed. For risky changes, a staging copy of production with its own database is the last boundary before users.
