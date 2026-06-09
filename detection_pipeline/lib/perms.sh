#!/bin/bash
# perms.sh — shared group-ownership + permission helpers for bucket outputs.
#
# Source AFTER pipeline.env so $OUTPUT_GROUP is set (falls back to reiteruni).
# Every operation is best-effort: you can only chgrp/chmod paths you own, so
# failures are swallowed and never abort the pipeline. Callers choose between:
#   - ensure_group_perms : silently make dirs group-owned + setgid (enforce)
#   - check_group_perms  : warn only, used as a startup preflight on a target
#                          dir that may also hold other users' files.

: "${OUTPUT_GROUP:=reiteruni}"

# ensure_group_perms <dir>...
#   chgrp each existing dir to $OUTPUT_GROUP and set mode 2775 (setgid + group
#   rwx) so new files created inside inherit the group and stay group-writable.
ensure_group_perms() {
	local d
	for d in "$@"; do
		[[ -e "$d" ]] || continue
		chgrp "$OUTPUT_GROUP" "$d" 2>/dev/null || true
		chmod 2775 "$d" 2>/dev/null || true
	done
}

# check_group_perms <dir>
#   Warn (return 1) if <dir> is not group=$OUTPUT_GROUP, not group-writable, or
#   not setgid. Never changes anything — meant for the target experiment dir,
#   which may contain other users' files we cannot chgrp.
check_group_perms() {
	local d="$1" warn=0 grp
	[[ -d "$d" ]] || { echo "[WARN] check_group_perms: not a directory: $d" >&2; return 1; }
	grp=$(stat -c '%G' "$d" 2>/dev/null || echo '?')
	if [[ "$grp" != "$OUTPUT_GROUP" ]]; then
		echo "[WARN] $d is group '$grp', expected '$OUTPUT_GROUP' — new outputs may not be group-shared." >&2
		warn=1
	fi
	if [[ -z "$(find "$d" -maxdepth 0 -perm -g+w 2>/dev/null)" ]]; then
		echo "[WARN] $d is not group-writable." >&2
		warn=1
	fi
	if [[ -z "$(find "$d" -maxdepth 0 -perm -2000 2>/dev/null)" ]]; then
		echo "[WARN] $d is not setgid — new files won't inherit the group." >&2
		warn=1
	fi
	if (( warn )); then
		echo "[HINT] fix the files you own under it:" >&2
		echo "       find '$d' -user \$USER ! -group $OUTPUT_GROUP -exec chgrp $OUTPUT_GROUP {} +" >&2
		echo "       find '$d' -user \$USER -type d -exec chmod g+rwxs {} +" >&2
		echo "       find '$d' -user \$USER -type f -exec chmod g+rw   {} +" >&2
	fi
	return $warn
}
