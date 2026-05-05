#!/usr/bin/env bash
set -Eeuo pipefail

# Usage:
#   ./local_push_and_test.sh "commit message" <student_id> [server_host] [server_project_dir]
#
# Example:
#   ./local_push_and_test.sh "optimize blackscholes" 123456789 10.26.200.25 ~/CSC3060_Project5
#
# Notes:
# - Run this on your LOCAL machine, from inside the git repo.
# - It will:
#   1) git add/commit/push
#   2) SSH to the course server
#   3) reset server repo to origin/main
#   4) rebuild in build-server
#   5) run ./run_stu and save output
# - The server reset is destructive to uncommitted server-side changes.

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 \"commit message\" <student_id> [server_host] [server_project_dir]"
  exit 1
fi

COMMIT_MSG="$1"
STUDENT_ID="$2"
SERVER_HOST="${3:-10.26.200.25}"
SERVER_PROJECT_DIR="${4:-~/CSC3060_Project5}"

echo "== Local git status =="
git status --short

echo "== Commit and push =="
git add .
if git diff --cached --quiet; then
  echo "No staged changes. Skipping commit."
else
  git commit -m "$COMMIT_MSG"
fi
git push origin main

echo "== Remote rebuild and test on server =="
ssh -p 2222 "${STUDENT_ID}@${SERVER_HOST}" bash -lc "'
set -Eeuo pipefail
cd ${SERVER_PROJECT_DIR}
git fetch origin
git reset --hard origin/main
rm -rf build-server
cmake -S . -B build-server
cmake --build build-server -j
cd build-server
./run_stu | tee run_stu_latest.txt
cp run_stu_latest.txt ../run_stu_latest.txt
echo
echo \"Saved result to: ${SERVER_PROJECT_DIR}/run_stu_latest.txt\"
'"
