#!/bin/bash -l
#SBATCH -t 2-0
#SBATCH -c 2
#SBATCH --partition=datacp
#SBATCH --mem=16G
#SBATCH -J backup
#SBATCH -o __JOBS_ROOT__/backup_%j.out
#SBATCH -e __JOBS_ROOT__/backup_%j.err
set -eo pipefail
umask 0002

JOBS_ROOT="__JOBS_ROOT__"
source "$JOBS_ROOT/pipeline.env"
source "$LIB_DIR/perms.sh"

if (( RUN_BACKUP != 1 )); then
	echo "[SKIP] backup disabled"
	exit 0
fi

command -v zip >/dev/null 2>&1 || { echo "[ERR] zip is not available on this node" >&2; exit 2; }

BACKUP_FILELIST="$JOBS_ROOT/backup_filelist.txt"
ARCHIVE_DIR="$(dirname "$BACKUP_ARCHIVE_PATH")"
LOCK_DIR="${BACKUP_ARCHIVE_PATH}.lock"

mkdir -p "$BACKUP_ROOT" "$ARCHIVE_DIR"
ensure_group_perms "$BACKUP_ROOT" "$ARCHIVE_DIR"

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
	echo "[ERR] backup archive is locked: $LOCK_DIR" >&2
	echo "      Another backup job may be updating $BACKUP_ARCHIVE_PATH." >&2
	exit 4
fi
cleanup_lock() {
	rmdir "$LOCK_DIR" 2>/dev/null || true
}
trap cleanup_lock EXIT

echo "[$(date)] building backup file list"
python3 "$LIB_DIR/backup_list.py" \
	--manifest "$JOBS_ROOT/manifest.csv" \
	--experiment-dir "$EXP_DIR" \
	--unit-root "$BACKUP_UNIT_ROOT" \
	--out "$BACKUP_FILELIST"

file_count=$(wc -l < "$BACKUP_FILELIST")
if (( file_count == 0 )); then
	echo "[ERR] backup file list is empty" >&2
	exit 2
fi

echo "[$(date)] updating backup archive: $BACKUP_ARCHIVE_PATH"
(
	cd "$BACKUP_UNIT_ROOT"
	if [[ -f "$BACKUP_ARCHIVE_PATH" ]]; then
		zip -0 -FS "$BACKUP_ARCHIVE_PATH" -@ < "$BACKUP_FILELIST"
	else
		zip -0 "$BACKUP_ARCHIVE_PATH" -@ < "$BACKUP_FILELIST"
	fi
	# Disable unzip's zip-bomb detection during the integrity test: it allocates
	# memory proportional to the archive and OOMs on multi-TB backups ("not enough
	# memory for bomb detection" -> spurious failure on block03's 1.9 TB zip). This
	# is our own archive, not untrusted input, so bomb detection adds no value.
	UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE zip -T "$BACKUP_ARCHIVE_PATH"
)

cat > "$BACKUP_DESC_PATH" <<EOF
name: $BACKUP_OWNER
project: $BACKUP_PROJECT

Automatic AntsArray pipeline raw-video backup.
source: $EXP_DIR
archive: $BACKUP_ARCHIVE_PATH
updated: $(date -Is)
files: $file_count

This is a stable per-block archive. Re-running the pipeline updates this same
local Backup archive; OIST's weekly Backup snapshots preserve older versions.
EOF

chgrp "$OUTPUT_GROUP" "$BACKUP_ARCHIVE_PATH" "$BACKUP_DESC_PATH" 2>/dev/null || true
chmod g+rw "$BACKUP_ARCHIVE_PATH" "$BACKUP_DESC_PATH" 2>/dev/null || true

echo "[OK] backup updated: $BACKUP_ARCHIVE_PATH ($file_count files)"
