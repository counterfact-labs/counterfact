# The supervisor

Long work runs detached (tmux, screen, or a job runner) so it survives the agent session ending. A supervisor is a plain shell script on a fixed interval, ten minutes works well, that moves each workstream one step forward. It costs nothing while it waits, and it calls a model only when there is work for one.

## State on disk

Each workstream (a "track") keeps its state as files, so any script or a new session can pick it up:

```
tracks/<track>/
  worktree        # path of the track's git worktree
  state/
    task-1.done     # one marker per finished step
    review-r2.md    # each review's full text
    attempts        # how many supervisor fix rounds have run
    pr-url          # written when the pull request opens
    BLOCKED         # written with a plain reason when the track needs the human
    DONE
```

Never edit a script while it runs; bash reads scripts as it goes. Log every milestone with a timestamp to one progress file the human and the next session can read.

## What one pass does

For each track that is neither DONE nor BLOCKED:

| Track state | Supervisor action |
|---|---|
| Review says FIX | Start a fix round from the findings, then review again |
| CI failed on the pull request | Start a fix round from the failing checks' logs |
| Pull request conflicts with the base | Merge the base branch in, rerun every check, review the merge diff |
| Green, reviewed, no merge watcher | Start the merge watcher |
| Third attempt on this track | Write BLOCKED with the reason and stop touching it |

## What it must never do

- Deploy, or run anything that costs money beyond its stated cap.
- Merge without every check passing, checked in the same command as the merge.
- Merge a change a user can see without the human's recorded approval (a label the merge checks for works well).
- Rewrite expected test results to make a test pass.
- Keep trying past three attempts. A stuck track gets a note, not a fourth round.

## Sketch

Adapt the commands to your tools; the structure is what matters.

```bash
#!/usr/bin/env bash
set -uo pipefail
ROOT=$HOME/agent-runs; LOG=$ROOT/progress.log
note() { echo "$(date '+%F %H:%M %Z') [supervisor] $*" >> "$LOG"; }

while :; do
  for t in "$ROOT"/tracks/*/; do
    s=$t/state; name=$(basename "$t")
    [ -e "$s/DONE" ] || [ -e "$s/BLOCKED" ] && continue
    pr=$(cat "$s/pr-url" 2>/dev/null) || continue          # no PR yet: the build script owns it
    n=$(cat "$s/attempts" 2>/dev/null || echo 0)
    checks=$(gh pr checks "$pr" --json bucket -q '[.[].bucket]|unique|join(",")' 2>/dev/null)
    case "$checks" in
      *pending*|"") continue ;;                              # wait for CI
      pass|pass,skipping|skipping,pass)
        [ -e "$s/merge-watcher" ] || { touch "$s/merge-watcher"; tmux new -d -s "merge-$name" "bash $ROOT/merge-on-green.sh $pr"; } ;;
      *)
        if [ "$n" -ge 3 ]; then echo "CI still failing after 3 fix rounds" > "$s/BLOCKED"; note "$name BLOCKED"; continue; fi
        echo $((n+1)) > "$s/attempts"; note "$name: fix round $((n+1))"
        bash "$ROOT/fix-round.sh" "$name" ;;                  # builder fixes, reviewer re-reviews, script commits
    esac
  done
  sleep 600
done
```

The merge watcher re-reads the checks immediately before merging and merges only if every one passed:

```bash
gh pr checks "$1" --json bucket -q 'all(.[]; .bucket=="pass" or .bucket=="skipping")' | grep -qx true \
  && gh pr merge "$1" --merge
```
