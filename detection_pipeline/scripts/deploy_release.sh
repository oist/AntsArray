#!/bin/bash
# deploy_release.sh — versioned cluster deploy of AntsArray to /apps/unit.
#
# Run on a deigo login by the maintainer. Produces:
#
#   <root>/releases/<YYYYMMDD>_<shortsha>/   immutable checkout of one commit
#   <root>/current -> releases/...           what users should invoke
#   <root>/.git-cache/                       bare mirror (fetch cache)
#
# Why releases instead of one updatable clone: pipeline.env bakes LIB_DIR /
# SCRIPTS_DIR / TEMPLATES_DIR absolute paths, and queued jobs re-read those
# files AT RUN TIME — a `git pull` under a live 3-day block would change its
# behaviour mid-run. A submitted wave resolves `current` with pwd -P and pins
# its release dir, so flipping the symlink never touches running work.
#
# Permissions: everything group-readable (reiteruni), directories setgid, and
# NOT group-writable — secondary users run the shared code, they must not be
# able to edit it.
#
# Usage:
#   deploy_release.sh [--root /apps/unit/ReiterU/AntsArray]
#                     [--repo-url <git url>] [--ref main] [--group reiteruni]
set -euo pipefail

ROOT=/apps/unit/ReiterU/AntsArray
REPO_URL=""
REF=main
GROUP=reiteruni

while [[ $# -gt 0 ]]; do
	case "$1" in
		--root) ROOT="$2"; shift 2 ;;
		--repo-url) REPO_URL="$2"; shift 2 ;;
		--ref) REF="$2"; shift 2 ;;
		--group) GROUP="$2"; shift 2 ;;
		-h|--help) sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
		*) echo "[ERR] unknown arg: $1" >&2; exit 2 ;;
	esac
done

CACHE="$ROOT/.git-cache"
mkdir -p "$ROOT/releases"

if [[ ! -d "$CACHE" ]]; then
	[[ -n "$REPO_URL" ]] || { echo "[ERR] first deploy needs --repo-url" >&2; exit 2; }
	git clone --bare "$REPO_URL" "$CACHE"
else
	git --git-dir="$CACHE" fetch origin "+refs/heads/*:refs/heads/*" --prune
fi

SHA=$(git --git-dir="$CACHE" rev-parse "$REF")
SHORT=${SHA:0:10}
REL="releases/$(date -u +%Y%m%d)_$SHORT"
DEST="$ROOT/$REL"

if [[ -d "$DEST" ]]; then
	echo "[INFO] $REL already deployed"
else
	# git archive -> tar: an immutable snapshot with no .git, so nobody can
	# mutate a release in place or accidentally commit from /apps/unit.
	mkdir -p "$DEST.tmp"
	git --git-dir="$CACHE" archive "$SHA" | tar -x -C "$DEST.tmp"
	printf 'sha=%s\nref=%s\ndate=%s\nby=%s\n' \
		"$SHA" "$REF" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$USER" > "$DEST.tmp/DEPLOY_INFO"
	# Group-readable, setgid dirs, never group-writable (see header).
	chgrp -R "$GROUP" "$DEST.tmp" 2>/dev/null \
		|| echo "[WARN] chgrp $GROUP failed; secondary users may not be able to read $DEST" >&2
	find "$DEST.tmp" -type d -exec chmod 2755 {} +
	find "$DEST.tmp" -type f -exec chmod 0644 {} +
	find "$DEST.tmp" -type f -name '*.sh' -exec chmod 0755 {} +
	mv "$DEST.tmp" "$DEST"
	echo "[INFO] deployed $REL"
fi

# ln -sfn replaces the symlink in one rename; running jobs hold the old
# target's real path (pwd -P) and are unaffected.
ln -sfn "$REL" "$ROOT/current"
echo "[INFO] current -> $REL ($SHA)"
echo "[INFO] users invoke: $ROOT/current/detection_pipeline/pipeline_multi.sh"
