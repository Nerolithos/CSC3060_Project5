# Quick GitHub + Server Test Scripts

These two scripts help you avoid repeated manual `push` / `pull` / rebuild / test steps.

## 1. local_push_and_test.sh
Run this on your local machine, inside the project repo.

Example:
```bash
chmod +x local_push_and_test.sh
./local_push_and_test.sh "optimize blackscholes" 124090302
```

What it does:
1. `git add .`
2. create a commit if there are staged changes
3. `git push origin main`
4. SSH to the course server
5. `git fetch origin && git reset --hard origin/main`
6. rebuild in `build-server`
7. run `./run_stu`

## 2. server_rebuild_and_test.sh
Run this directly on the course server if you have already pushed and only want to retest.

Example:
```bash
chmod +x server_rebuild_and_test.sh
./server_rebuild_and_test.sh
```

## Important
Both scripts use:

```bash
git reset --hard origin/main
```

So they will discard uncommitted changes in the server-side repo.

If you often keep unfinished edits on the server, do not use these scripts without checking first.
