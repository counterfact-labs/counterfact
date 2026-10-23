---
name: running-a-team-of-coding-agents
description: Use when one person directs several coding agents (Claude Code, Codex, or others) on production software, runs parallel workstreams or long unattended runs, or when agent work keeps failing review on the same rule, drifts in structure, breaks at integration boundaries, or costs more than expected.
---

# Running a Team of Coding Agents

## Overview

The practices are the established ones: specs, tests, code review, integration testing, staging, modular design, and regular refactoring. What changes is the material. Agents forget everything between sessions, follow instructions literally, and write code faster than anyone can read it, so each practice has to live in files the agents read and in scripts that enforce it, rather than in anyone's memory.

The human keeps money, product direction, production risk, deploys, and every change a user can see. Everything else can be delegated, provided the delegation is written down.

## When to use

- You coordinate more than one agent, or one agent across sessions, on code that real users depend on.
- Work runs while you are away (overnight builds, fix rounds, merge waits).
- A feature fails review three times, the same decision appears in several modules, or bugs pass every check and still break in production.
- Spend is high and most of it is agents rereading context.

Not for a one-off script or a throwaway prototype; the overhead only pays for itself on code you will maintain.

## The practice

**1. Specs, plans, and tests first.** Each feature goes from brainstorm to written spec, to a plan of small tasks with tests, to a test-first build, to a review against the spec. Every logic module has its own test file and new code is fully covered. Without that rule, agents skip tests for the code that is awkward to test, which is usually the code that breaks. (The superpowers plugin implements this flow; use it where it is installed.)

**2. Decisions as multiple-choice questions, one at a time.** Ask as many questions as the work needs, with no cap. Each question gives options written as the actual outcome, the recommended one first and labelled. Decide technical details yourself; bring the person subjective calls and tradeoffs (what a screen shows, how much risk is acceptable), because they are accountable for them.

**3. Roles, worktrees, and a supervisor.** One agent builds from short written briefs, one plans, writes briefs, and reads every diff before committing, and a strong model reviews every task and the whole branch before a pull request opens. Every task gets its own git worktree and branch; never share a checkout between agents. Long runs live in tmux (or another detached runner) with state markers on disk, and a supervisor script checks each workstream on a fixed interval. See `reference/roles-and-briefs.md` and `reference/supervisor.md`.

**4. Rules written after failures.** Agents remember nothing between sessions, so every lesson goes into a file they read before starting (AGENTS.md, CLAUDE.md, or the brief template). One incident is enough to justify a rule. New integrations get at least one test against the real service, in a test mode or sandbox.

**5. Structure audits and one owner per decision.** Agents add a condition wherever they happen to be looking, so the same decision spreads across modules. Each decision has exactly one owning module that everything else calls. After two failed fix rounds on the same rule, stop fixing and consolidate the rule in its owner first, as its own change. Run structure audits as separate work, and treat repeated review failures as a sign one is overdue. This is information hiding (Parnas, 1972) applied to a codebase whose authors cannot see each other's work.

**6. Integration tests and staging.** Bugs that pass type checks, lint, unit tests, and review tend to sit at boundaries: data only a seed script creates, a page built without runtime settings, a tool whose structured output is missing the payload. Give integration tests the same weight as unit tests, and pass risky changes through a staging copy of production with its own database. See `reference/rules-and-audits.md`.

**7. Cost.** Most spend pays for agents rereading context. Keep briefs narrow, have reviewers read diffs rather than whole files, use the cheaper builder for building, and wait in shell scripts, which cost nothing, rather than in agent sessions.

**8. What stays with the human.** Money, product direction, production risk, deploys, and user-visible changes. Prepare these as one-step decisions (a script to run, an approval to give) and never take them on the person's behalf.

## Quick reference

| Situation | Do this |
|---|---|
| Starting a task | New worktree and branch; brief names the files, the tests, and what not to touch |
| Review fails | Fix round from the review's findings only; rerun review on the new diff |
| Second failed round on the same rule | Stop; find every place that answers the question; move it to one owner first |
| Third attempt on any workstream | Supervisor stops and leaves a note for the human |
| PR conflicts with the base branch | Merge the base in, rerun every check, then review the merge diff |
| PR is green and reviewed | Merge in plan order, gating the merge on the check result in the same command |
| Change is visible to users | Screenshot before and after; wait for explicit approval of that change |
| Deploy is ready | Write the deploy script; the human runs it |
| Open decisions | One multiple-choice question at a time, recommendation first |

## Red flags

These thoughts mean stop and follow the rule instead.

| Thought | Reality |
|---|---|
| "One more fix round will get it" | Two failed rounds on one rule mean the rule lives in several places, so consolidate it in one owner first. |
| "They said ship it, so I can deploy" | Deploys stay with the human every time, and an earlier approval does not cover the next deploy. |
| "I'll start the second task in this checkout" | A shared checkout lets one agent commit another's half-finished work, so start a new worktree. |
| "All the checks passed" | Checks cover the parts; test the boundaries with the real service, real build settings, and real data. |
| "I'll ask the three most important questions" | Ask every genuine question, one at a time, and decide the technical details yourself. |
| "It's a small wording change" | Every user-visible change needs the human's approval. |
| "The reviewer didn't return a verdict" | Treat a missing verdict as a failed review and run the review again. |

## Reference files

- `reference/roles-and-briefs.md`: role split, brief template, review prompt, and cost habits.
- `reference/supervisor.md`: supervisor loop, state markers, merge watcher, and what it must never do.
- `reference/rules-and-audits.md`: seed rules for AGENTS.md, writing rules after incidents, structure audits, and boundary tests.
