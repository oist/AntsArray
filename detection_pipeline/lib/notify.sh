# Best-effort email notification helper.
# source from login-side watchers or sbatch jobs after NOTIFY_EMAIL is set:
#   source "$LIB_DIR/notify.sh"
#   notify_send "subject" "body"
#
# No-op when NOTIFY_EMAIL is empty, and ALWAYS returns 0: every caller runs
# under set -e, and a broken mailer must never abort a pipeline stage.
#
# Delivery cascade: mail -> mailx -> sendmail -> a zero-work sbatch job whose
# Slurm END mail carries the subject as the job name. The sbatch fallback is
# what makes this work on compute nodes without a local MTA (anywhere sbatch
# works, Slurm's own MailProg can deliver) -- but it drops the body, so keep
# subjects specific enough to stand alone.

notify_send() {
	local subject="$1" body="${2:-$1}"
	[[ -n "${NOTIFY_EMAIL:-}" ]] || return 0

	if command -v mail >/dev/null 2>&1; then
		if printf '%s\n' "$body" | mail -s "$subject" "$NOTIFY_EMAIL" 2>/dev/null; then
			echo "[notify] emailed (mail): $subject"
			return 0
		fi
	fi
	if command -v mailx >/dev/null 2>&1; then
		if printf '%s\n' "$body" | mailx -s "$subject" "$NOTIFY_EMAIL" 2>/dev/null; then
			echo "[notify] emailed (mailx): $subject"
			return 0
		fi
	fi

	local sendmail_bin=""
	if command -v sendmail >/dev/null 2>&1; then
		sendmail_bin="sendmail"
	elif [[ -x /usr/sbin/sendmail ]]; then
		sendmail_bin="/usr/sbin/sendmail"
	fi
	if [[ -n "$sendmail_bin" ]]; then
		if printf 'To: %s\nSubject: %s\n\n%s\n' "$NOTIFY_EMAIL" "$subject" "$body" \
				| "$sendmail_bin" -t 2>/dev/null; then
			echo "[notify] emailed (sendmail): $subject"
			return 0
		fi
	fi

	if command -v sbatch >/dev/null 2>&1; then
		# Neither deigo nor saion has a system default partition, so try likely
		# ones in order (deigo: short/compute/datacp; saion: test-gpu/datacp).
		local part
		for part in ${NOTIFY_SBATCH_PARTITIONS:-short compute test-gpu datacp}; do
			if sbatch --parsable --job-name="$subject" \
					--mail-type=END --mail-user="$NOTIFY_EMAIL" \
					--partition="$part" -t 0-0:02 -c 1 --mem=100M -o /dev/null -e /dev/null \
					--wrap=true >/dev/null 2>&1; then
				echo "[notify] emailed via Slurm END-mail job on '$part' (subject only): $subject"
				return 0
			fi
		done
	fi

	echo "[notify] WARNING: no mailer available; NOT emailed: $subject" >&2
	return 0
}
