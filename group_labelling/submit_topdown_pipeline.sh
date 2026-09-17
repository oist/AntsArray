#!/bin/bash -l
# submit_topdown_pipeline.sh -- one driver for the whole top-down loop on saion.
#
# The operational loop, which repeats every time a review batch lands:
#
#   sync      copy this directory's python to $SCRIPTS (see DEPLOY below)
#   flatten   cut the source_video chains so sleap_io can read the packages at all
#   arms      split into camera-disjoint training arms + a frozen test set
#   configs   write one warm-started training config per arm
#   train     one GPU job per arm                                    [GPU, SLURM]
#   predict   ground-truth-anchored inference on the unreviewed chunks [GPU, SLURM]
#   select    build the next review package + order CSV
#   extract   cut the finished frames out of a part-reviewed package
#   eval      predict each trained arm on the test set, then per-node error
#             and the body-frame convention check                    [GPU, SLURM]
#
# The offline A/B of selection rules is the same machinery on different inputs and is
# NOT part of the loop -- see simulate_review_ch06.py for what it has already settled
# and why it is kept:
#
#   ab-score  GT-anchored predictions on collaborator ch06, from a LEAK-FREE model
#   ab-arms   one training package per (strategy, budget), plus floor and ceiling
#   ab-configs / ab-train / ab-eval   as above, on the A/B arms
#
# Usage:
#   bash submit_topdown_pipeline.sh --stages sync,flatten,arms,configs
#   bash submit_topdown_pipeline.sh --stages train        # then wait for squeue
#   bash submit_topdown_pipeline.sh --stages predict      # then wait
#   bash submit_topdown_pipeline.sh --stages select
#
# DEPLOY. Every python entry point imports topdown_common.py by module name, so
# $SCRIPTS must hold it alongside them or every stage dies on ImportError. The `sync`
# stage copies the whole set; a preflight refuses to run python stages without it.
#
# Filesystem rules on this cluster, learned the hard way:
#   * /bucket is READ-ONLY on compute nodes. Anything a job writes must go to
#     /work/ReiterU (writable) or node-local /scratch. Stages that only write /bucket
#     run on the login node; the rest run under SLURM.
#   * saion-gpu25 cannot resolve the login host, so its jobs report COMPLETE while
#     their uploads and logs vanish. It is excluded everywhere.
#   * The account's largegpu association caps at cpu=128, gres/gpu=8, mem=1T, so
#     16 cores + 128G is exactly the slice that lets all 8 GPUs run concurrently.
#     Over-requesting memory silently costs GPU slots.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SLEAP_NN_VERSION="${SLEAP_NN_VERSION:-0.3.1}"
PY="/apps/unit/ReiterU/sleap-nn/${SLEAP_NN_VERSION}/tools/sleap-nn/bin/python"
SLEAP_NN_TOOL_BIN="/apps/unit/ReiterU/sleap-nn/${SLEAP_NN_VERSION}/tools/sleap-nn/bin"
SLEAP_NN_BIN="/apps/unit/ReiterU/sleap-nn/${SLEAP_NN_VERSION}/bin"

BUCKET="${BUCKET:-/bucket/ReiterU/Ants/SLEAP_files/Group_labelling/20260803_topdown}"
WORK="${WORK:-/work/ReiterU/Ants/topdown_20260803}"
SCRIPTS="${SCRIPTS:-$HOME/topdown/scripts}"
LOGS="${LOGS:-$HOME/topdown_logs}"
MANIFEST="${MANIFEST:-$BUCKET/chunk_manifest.csv}"

MODELS="${MODELS:-/bucket/ReiterU/Ants/SLEAP_files/Simple_skeleton_nn}"
CENTROID_MODEL="${CENTROID_MODEL:-$MODELS/250408_141245.centroid}"
INSTANCE_MODEL="${INSTANCE_MODEL:-$MODELS/250408_141245.centered_instance}"

# Scoring model for the review selection. reviewed260 is the only centered-instance
# model trained purely on reviewed labels, so it embodies the antenna convention the
# review exists to propagate, and it never saw a ch01-05 camera -- which is what makes
# its disagreement there a measure of difficulty rather than of memorisation.
MODEL="${MODEL:-$WORK/models_reviewed260/reviewed260_centered_instance/reviewed260_centered_instance}"
PRED_DIR="${PRED_DIR:-$WORK/review_selection/pred}"
OUT_DIR="${OUT_DIR:-$BUCKET/review_ch01to05}"
EVAL_DIR="${EVAL_DIR:-$WORK/eval}"

CHUNKS="${CHUNKS:-1,2,3,4,5}"
SELECT_MODE="${SELECT_MODE:-frame}"
MIN_FRAME_DENSITY="${MIN_FRAME_DENSITY:-16}"
TOTAL_BUDGET="${TOTAL_BUDGET:-}"          # empty => the selector's own default
PER_CAMERA_QUOTA="${PER_CAMERA_QUOTA:-40}"
SELECT_STEM="${SELECT_STEM:-ch01to05_dense}"

# extract stage
REVIEWED_SLP="${REVIEWED_SLP:-}"
ORIGINAL_SLP="${ORIGINAL_SLP:-}"
EXTRACT_OUT="${EXTRACT_OUT:-$WORK/reviewed_batches/reviewed_batch.pkg.slp}"

# eval stage
TEST_SLP="${TEST_SLP:-$(ls "$BUCKET"/arms_cam24/*_REVIEWED_test.pkg.slp 2>/dev/null | head -1 || true)}"
CONFIGS_JSON="${CONFIGS_JSON:-$BUCKET/cfg/configs.json}"
CKPT_ROOT="${CKPT_ROOT:-$WORK/models}"

# Frames per batch, NOT crops. Ground-truth anchoring turns every instance in a
# batched frame into its own 640px crop, so a dense camera at 30 ants/frame makes this
# 30x larger than it looks. Above ~400 crops the decoder's bilinear upsample overflows
# INT_MAX and the job dies within seconds.
PRED_BATCH="${PRED_BATCH:-4}"

PARTITION="${PARTITION:-largegpu}"
WALLTIME="${WALLTIME:-12:00:00}"
EXCLUDE="${EXCLUDE:-saion-gpu25}"
CPUS="${CPUS:-16}"
MEM="${MEM:-128G}"

# --- offline A/B --------------------------------------------------------------
AB="${AB:-$WORK/selection_ab}"
SCORE_MODEL="${SCORE_MODEL:-$WORK/models_localclone/ci_A2_ch01to05/ci_A2_ch01to05}"
COLLAB="${COLLAB:-$(ls "$BUCKET"/flat/*chunk06*recorrected.pkg.slp 2>/dev/null | head -1 || true)}"
REVIEWED_CH06="${REVIEWED_CH06:-$(ls "$BUCKET"/flat/*chunk06*_REVIEWED.slp 2>/dev/null | head -1 || true)}"
HOLDOUT_CAMERA="${HOLDOUT_CAMERA:-cam24}"
STRATEGIES="${STRATEGIES:-random,disagreement,lowconf,diverse,oracle}"
BUDGETS="${BUDGETS:-600}"
# All arms share one fixed seed. The deployed rev200 config left `seed: null`, which is
# fine for a single production run and useless here: run-to-run variance would be
# indistinguishable from a strategy effect.
SEED="${SEED:-42}"
AB_BATCH="${AB_BATCH:-6}"
AB_LR="${AB_LR:-0.0001}"
AB_MAX_EPOCHS="${AB_MAX_EPOCHS:-200}"
AB_EARLY_PATIENCE="${AB_EARLY_PATIENCE:-80}"
AB_PLATEAU_PATIENCE="${AB_PLATEAU_PATIENCE:-5}"
AB_VAL_FRACTION="${AB_VAL_FRACTION:-0.35}"
AB_NUM_WORKERS="${AB_NUM_WORKERS:-12}"

CORE_SCRIPTS=(
	topdown_common.py
	flatten_slp_videos.py
	build_topdown_arms.py
	make_topdown_configs.py
	predict_gt_anchored.py
	select_review_instances.py
	extract_reviewed_batch.py
	score_node_error.py
	simulate_review_ch06.py
)

DRY_RUN=0
STAGES="all"
while [[ $# -gt 0 ]]; do
	case "$1" in
		--dry-run) DRY_RUN=1; shift ;;
		--stages)  STAGES="$2"; shift 2 ;;
		-h|--help) sed -n '2,44p' "$0"; exit 0 ;;
		*) echo "[ERR] unknown arg: $1" >&2; exit 2 ;;
	esac
done

runs_stage() { [[ "$STAGES" == "all" || ",$STAGES," == *",$1,"* ]]; }

STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGS" "$SCRIPTS" "$PRED_DIR" "$EVAL_DIR"
[[ -x "$PY" ]] || { echo "[ERR] no sleap-nn $SLEAP_NN_VERSION python at $PY" >&2; exit 3; }

# ------------------------------------------------------------------- helpers

# Write an sbatch file from stdin, wrapped in the standard largegpu header, and submit
# it. One definition, so every job on this cluster gets the same allocation, the same
# node exclusion and the same log naming.
gpu_sbatch() {
	local name="$1" walltime="$2" logstem="$3"
	local file="$SCRIPTS/${name}.sbatch"
	{
		echo "#!/bin/bash -l"
		echo "#SBATCH --job-name=${name}"
		echo "#SBATCH --partition=${PARTITION}"
		echo "#SBATCH --gres=gpu:a100:1"
		echo "#SBATCH --cpus-per-task=${CPUS}"
		echo "#SBATCH --mem=${MEM}"
		echo "#SBATCH --time=${walltime}"
		echo "#SBATCH --exclude=${EXCLUDE}"
		echo "#SBATCH --output=${LOGS}/${logstem}_%j.log"
		echo "set -euo pipefail"
		echo "export TMPDIR=/scratch"
		echo 'echo "[$(date -Is)] $(hostname)"'
		cat
	} > "$file"
	if (( DRY_RUN )); then
		echo "[dry-run] would submit $file"
		return 0
	fi
	local jid
	jid=$(sbatch --parsable "$file")
	echo "[OK] $name -> job $jid"
	echo "     log: $LOGS/${logstem}_${jid}.log"
}

# --dry-run must never mutate anything, login-side stages included.
run() {
	if (( DRY_RUN )); then
		echo "[dry-run] $*"
		return 0
	fi
	"$@"
}

arms_of() {   # configs.json -> "name<TAB>config<TAB>ckpt_dir" per line
	"$PY" -c "
import json, sys
for name, spec in json.load(open(sys.argv[1])).items():
    print(name, spec['config'], spec['ckpt_dir'], sep='\t')
" "$1"
}

train_arms() {   # configs.json, job-name prefix
	local configs="$1" prefix="$2" row name cfg ckpt_dir
	[[ -f "$configs" ]] || { echo "[ERR] no $configs; run the configs stage" >&2; exit 6; }
	mapfile -t ARMS < <(arms_of "$configs")
	(( ${#ARMS[@]} )) || { echo "[ERR] no arms in $configs" >&2; exit 7; }
	for row in "${ARMS[@]}"; do
		IFS=$'\t' read -r name cfg ckpt_dir <<<"$row"
		gpu_sbatch "${prefix}_${name}" "$WALLTIME" "${prefix}_${name}_${STAMP}" <<EOF
export PATH="${SLEAP_NN_BIN}:\$PATH"
mkdir -p "${ckpt_dir}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
sleap-nn train --config "${cfg}"
# /bucket is read-only here, so publishing back is left to the login-side rsync.
touch "${ckpt_dir}/.training_complete"
EOF
	done
}

# Predict with every trained arm on one test set, then score it. Scoring is pure numpy
# and rides along on the same compute node rather than needing a third hop.
score_arms() {   # configs.json, test .slp, ckpt root, eval dir
	local configs="$1" test_slp="$2" ckpt_root="$3" eval_dir="$4"
	local test_stem name ckpt pred
	test_stem="$(basename "$test_slp" .pkg.slp)"
	mkdir -p "$eval_dir"
	while IFS=$'\t' read -r name _ _; do
		ckpt="$ckpt_root/$name/$name"
		[[ -f "$ckpt/best.ckpt" ]] || { echo "[!!] $name: no best.ckpt yet, skipped"; continue; }
		pred="$eval_dir/$name.pred.slp"
		if [[ ! -f "$pred" ]]; then
			"$PY" -u "$SCRIPTS/predict_gt_anchored.py" --model "$ckpt" \
				--out-dir "$eval_dir" --staging-dir "$eval_dir/model_$name" \
				--sleap-nn-bin "$SLEAP_NN_TOOL_BIN" --batch-size "$PRED_BATCH" \
				--device "${EVAL_DEVICE:-cuda}" "$test_slp" >/dev/null
			mv "$eval_dir/$test_stem.pred.slp" "$pred"
		fi
		"$PY" -u "$SCRIPTS/score_node_error.py" --gt "$test_slp" --pred "$pred" \
			--label "$name" --report "$eval_dir/$name.json" | tail -12
	done < <(arms_of "$configs")
}

eval_summary() {   # eval dir [, arms.json for the corrected counts]
	"$PY" -c "
import json, glob, sys
eval_dir = sys.argv[1]
arms_json = sys.argv[2] if len(sys.argv) > 2 else ''
rows = []
for path in sorted(glob.glob(eval_dir + '/*.json')):
    doc = json.load(open(path))
    base = doc.get('antenna_base') or {}
    if not base.get('n'):
        continue
    rows.append((doc['label'], base['median'], base['mean'], base.get('pck@5') or 0.0,
                 (doc.get('all_other_nodes') or {}).get('median') or 0.0))
arms = {}
if arms_json:
    try:
        arms = json.load(open(arms_json)).get('arms', {})
    except OSError:
        pass
floor = next((r for r in rows if r[0] == 'floor_n0'), None)
print(f\"{'arm':<24}{'corrected':>10}{'base med':>10}{'base mean':>11}{'pck@5':>8}{'other med':>11}{'vs floor':>10}\")
print('-' * 84)
for r in sorted(rows, key=lambda r: r[1]):
    n = arms.get(r[0], {}).get('n_corrected', '')
    delta = ('%+.2f' % (r[1] - floor[1])) if floor else ''
    print(f'{r[0]:<24}{str(n):>10}{r[1]:>10.2f}{r[2]:>11.2f}{r[3]:>8.3f}{r[4]:>11.2f}{delta:>10}')
print()
print('base med = median antenna-base error on the held-out cameras; lower is better.')
" "$@"
}

need_scripts() {
	local missing=() f
	for f in "${CORE_SCRIPTS[@]}"; do
		[[ -f "$SCRIPTS/$f" ]] || missing+=("$f")
	done
	if (( ${#missing[@]} )); then
		echo "[ERR] $SCRIPTS is missing: ${missing[*]}" >&2
		echo "      every entry point imports topdown_common by module name." >&2
		echo "      run: bash $0 --stages sync" >&2
		exit 3
	fi
}

# --------------------------------------------------------------------- sync
if runs_stage sync; then
	echo "[sync] $HERE -> $SCRIPTS"
	if (( DRY_RUN )); then
		echo "[dry-run] would copy ${CORE_SCRIPTS[*]}"
	else
		for f in "${CORE_SCRIPTS[@]}"; do
			cp "$HERE/$f" "$SCRIPTS/$f"
		done
		cp "$HERE/$(basename "$0")" "$SCRIPTS/"
		echo "[OK] ${#CORE_SCRIPTS[@]} module(s) + the driver copied"
	fi
fi

# ------------------------------------------------------------------ flatten
if runs_stage flatten; then
	need_scripts
	if [[ -f "$BUCKET/flat/flatten_report.json" ]]; then
		echo "[skip] flatten: $BUCKET/flat already built"
	else
		echo "[flatten] cutting source_video chains"
		run "$PY" "$SCRIPTS/flatten_slp_videos.py" --out-dir "$BUCKET/flat" \
			--report "$BUCKET/flat/flatten_report.json" "$BUCKET"/raw/*.pkg.slp
		# The reviewed export carries no images; bind it to the package that has them.
		reviewed=$(ls "$BUCKET"/raw/*_recorrected.slp 2>/dev/null | head -1 || true)
		if [[ -n "$reviewed" ]]; then
			run "$PY" "$SCRIPTS/flatten_slp_videos.py" \
				--out-dir "$BUCKET/flat" --suffix _REVIEWED \
				--images-from "$BUCKET/flat/$(basename "${reviewed%.slp}.pkg.slp")" "$reviewed"
		fi
	fi
fi

# --------------------------------------------------------------------- arms
if runs_stage arms; then
	need_scripts
	if [[ -f "$BUCKET/arms/arms.json" ]]; then
		echo "[skip] arms: $BUCKET/arms/arms.json already built"
	else
		echo "[arms] building camera-disjoint arms"
		extra=()
		[[ -n "${EXTRA_ARM:-}" ]] && extra=(--extra-arm "$EXTRA_ARM")
		run "$PY" "$SCRIPTS/build_topdown_arms.py" --flat-dir "$BUCKET/flat" \
			--manifest "$MANIFEST" --out-dir "$BUCKET/arms" \
			"${extra[@]}" --report "$BUCKET/arms/arms.json"
	fi
fi

# ------------------------------------------------------------------ configs
if runs_stage configs; then
	need_scripts
	echo "[configs] writing warm-started configs"
	run "$PY" "$SCRIPTS/make_topdown_configs.py" \
		--arms "$BUCKET/arms/arms.json" --out-dir "$BUCKET/cfg" \
		--centroid-model "$CENTROID_MODEL" --instance-model "$INSTANCE_MODEL" \
		--ckpt-root "$CKPT_ROOT" --report "$CONFIGS_JSON"
fi

# -------------------------------------------------------------------- train
if runs_stage train; then
	need_scripts
	echo "[train] one job per arm"
	train_arms "$CONFIGS_JSON" td
	echo "Watch:   squeue -u \$USER"
	echo "Publish: rsync -a $CKPT_ROOT/ $BUCKET/models/   # run on the login node"
fi

# ------------------------------------------------------------------ predict
if runs_stage predict; then
	need_scripts
	mapfile -t INPUTS < <(ls "$BUCKET"/flat/*chunk0[1-5]*.pkg.slp 2>/dev/null)
	(( ${#INPUTS[@]} )) || { echo "[ERR] no ch01-05 packages in $BUCKET/flat" >&2; exit 4; }
	[[ -f "$MODEL/best.ckpt" ]] || { echo "[ERR] no best.ckpt in $MODEL" >&2; exit 3; }

	missing=0
	for src in "${INPUTS[@]}"; do
		[[ -f "$PRED_DIR/$(basename "$src" .pkg.slp).pred.slp" ]] || missing=1
	done
	if (( ! missing )); then
		echo "[skip] predict: $PRED_DIR already complete"
	else
		echo "[predict] $(basename "$MODEL") vs ch01-05, ground-truth anchored"
		gpu_sbatch td_predict "${PREDICT_WALLTIME:-04:00:00}" "predict_${STAMP}" <<EOF
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -u "$SCRIPTS/predict_gt_anchored.py" \\
	--model "$MODEL" --out-dir "$PRED_DIR" \\
	--sleap-nn-bin "$SLEAP_NN_TOOL_BIN" --batch-size "$PRED_BATCH" \\
	--report "$PRED_DIR/predict_report.json" \\
	${INPUTS[@]}
EOF
		echo "     then: bash $0 --stages select"
		(( DRY_RUN )) || exit 0
	fi
fi

# ------------------------------------------------------------------- select
if runs_stage select; then
	need_scripts
	echo "[select] building the review package"
	stem="${SELECT_STEM}${TOTAL_BUDGET:+_n$TOTAL_BUDGET}"
	log="$LOGS/select_${STAMP}.log"
	cmd=("$PY" -u "$SCRIPTS/select_review_instances.py"
		--flat-dir "$BUCKET/flat" --pred-dir "$PRED_DIR" --manifest "$MANIFEST"
		--chunks "$CHUNKS" --mode "$SELECT_MODE"
		--min-frame-density "$MIN_FRAME_DENSITY"
		--per-camera-quota "$PER_CAMERA_QUOTA"
		--out "$OUT_DIR/$stem.pkg.slp" --report "$OUT_DIR/$stem.json")
	[[ -n "$TOTAL_BUDGET" ]] && cmd+=(--total-budget "$TOTAL_BUDGET")
	if (( DRY_RUN )); then
		echo "[dry-run] ${cmd[*]}"
	else
		mkdir -p "$OUT_DIR"
		"${cmd[@]}" 2>&1 | tee "$log"
		echo "[OK] log -> $log"
	fi
fi

# ------------------------------------------------------------------ extract
if runs_stage extract; then
	need_scripts
	[[ -n "$REVIEWED_SLP" && -n "$ORIGINAL_SLP" ]] || {
		echo "[ERR] set REVIEWED_SLP and ORIGINAL_SLP for the extract stage" >&2; exit 4; }
	echo "[extract] cutting the finished frames out of $(basename "$REVIEWED_SLP")"
	log="$LOGS/extract_${STAMP}.log"
	cmd=("$PY" -u "$SCRIPTS/extract_reviewed_batch.py"
		--reviewed "$REVIEWED_SLP" --original "$ORIGINAL_SLP"
		--out "$EXTRACT_OUT" --report "${EXTRACT_OUT%.pkg.slp}.json")
	if (( DRY_RUN )); then
		echo "[dry-run] ${cmd[*]}"
	else
		"${cmd[@]}" 2>&1 | tee "$log"
		echo "[OK] log -> $log"
		echo "     next: --stages arms with EXTRA_ARM=<name>=$EXTRACT_OUT"
	fi
fi

# --------------------------------------------------------------------- eval
# Predicting with every arm needs a GPU and the login node has none, so this stage
# submits itself and re-enters with EVAL_INLINE=1 on the compute node.
if runs_stage eval && [[ "${EVAL_INLINE:-0}" != "1" ]]; then
	need_scripts
	[[ -n "$TEST_SLP" && -f "$TEST_SLP" ]] || { echo "[ERR] no TEST_SLP: '$TEST_SLP'" >&2; exit 4; }
	echo "[eval] submitting the scoring pass"
	gpu_sbatch td_eval "${EVAL_WALLTIME:-04:00:00}" "eval_${STAMP}" <<EOF
EVAL_INLINE=1 TEST_SLP="$TEST_SLP" CONFIGS_JSON="$CONFIGS_JSON" \\
	CKPT_ROOT="$CKPT_ROOT" EVAL_DIR="$EVAL_DIR" SCRIPTS="$SCRIPTS" \\
	bash "$SCRIPTS/$(basename "$0")" --stages eval
EOF
	(( DRY_RUN )) || exit 0
fi

if runs_stage eval && [[ "${EVAL_INLINE:-0}" == "1" ]]; then
	echo "[eval] scoring every arm on $(basename "$TEST_SLP")"
	score_arms "$CONFIGS_JSON" "$TEST_SLP" "$CKPT_ROOT" "$EVAL_DIR"
	echo
	eval_summary "$EVAL_DIR" | tee "$LOGS/eval_summary_${STAMP}.log"
fi

# ===================================================== offline A/B (not the loop)

if runs_stage ab-score || runs_stage ab-arms || runs_stage ab-configs \
	|| runs_stage ab-train || runs_stage ab-eval; then
	need_scripts
	mkdir -p "$AB"/{pred,arms,cfg,models,eval}
	AB_SCORE_PRED="$AB/pred/$(basename "${COLLAB:-none}" .pkg.slp).pred.slp"
	AB_CONFIGS="$AB/cfg/configs.json"
fi

if runs_stage ab-score; then
	# The leak check comes FIRST, before any file test. reviewed260 and every arm that
	# trained on ch06 have memorised exactly the corrections the `disagreement`
	# strategy would be ranked on, so they cannot score this experiment. A missing
	# checkpoint is a typo; a leaking scorer silently invalidates the whole result,
	# so it must be refused whether or not its files happen to be in place.
	case "$SCORE_MODEL" in
		*reviewed260*|*rev200*|*ch06full*|*A1*|*A3*|*A4*|*A5*|*final_v2*)
			echo "[ERR] SCORE_MODEL=$SCORE_MODEL was trained on ch06 and would leak the" >&2
			echo "      corrections into the ranking. Use a model that never saw ch06." >&2
			exit 5 ;;
	esac
	for f in "$COLLAB" "$REVIEWED_CH06" "$TEST_SLP"; do
		[[ -f "$f" ]] || { echo "[ERR] missing A/B input: '$f'" >&2; exit 4; }
	done
	[[ -f "$SCORE_MODEL/best.ckpt" ]] || { echo "[ERR] no best.ckpt in $SCORE_MODEL" >&2; exit 4; }
	if [[ -f "$AB_SCORE_PRED" ]]; then
		echo "[skip] ab-score: $AB_SCORE_PRED exists"
	else
		echo "[ab-score] scorer=$(basename "$SCORE_MODEL") holdout=$HOLDOUT_CAMERA seed=$SEED"
		gpu_sbatch ab_score "02:00:00" "ab_score_${STAMP}" <<EOF
"$PY" -u "$SCRIPTS/predict_gt_anchored.py" --model "$SCORE_MODEL" \\
	--out-dir "$AB/pred" --staging-dir "$AB/pred/model_scorer" \\
	--sleap-nn-bin "$SLEAP_NN_TOOL_BIN" --batch-size "$PRED_BATCH" \\
	--report "$AB/pred/score_report.json" "$COLLAB"
EOF
		echo "     then: bash $0 --stages ab-arms,ab-configs,ab-train"
		(( DRY_RUN )) || exit 0
	fi
fi

if runs_stage ab-arms; then
	if [[ -f "$AB/arms/arms.json" ]]; then
		echo "[skip] ab-arms: $AB/arms/arms.json exists"
	else
		[[ -f "$AB_SCORE_PRED" ]] || {
			echo "[ERR] no $AB_SCORE_PRED; run --stages ab-score" >&2; exit 6; }
		echo "[ab-arms] one training package per (strategy, budget)"
		run "$PY" -u "$SCRIPTS/simulate_review_ch06.py" \
			--collab "$COLLAB" --reviewed "$REVIEWED_CH06" --pred "$AB_SCORE_PRED" \
			--manifest "$MANIFEST" --test-slp "$TEST_SLP" \
			--holdout-camera "$HOLDOUT_CAMERA" --strategies "$STRATEGIES" \
			--budgets "$BUDGETS" --seed "$SEED" \
			--out-dir "$AB/arms" --arms-json "$AB/arms/arms.json" \
			--report "$AB/arms/records.json" 2>&1 | tee "$LOGS/ab_arms_${STAMP}.log"
	fi
fi

if runs_stage ab-configs; then
	echo "[ab-configs] one config per arm, identical but for the training package"
	run "$PY" -u "$SCRIPTS/make_topdown_configs.py" \
		--arms "$AB/arms/arms.json" --out-dir "$AB/cfg" \
		--centroid-model "$CENTROID_MODEL" --instance-model "$INSTANCE_MODEL" \
		--batch-centroid "$AB_BATCH" --batch-instance "$AB_BATCH" \
		--ckpt-root "$AB/models" --cache-root /scratch/ab_cache \
		--max-epochs "$AB_MAX_EPOCHS" --early-stopping-patience "$AB_EARLY_PATIENCE" \
		--plateau-patience "$AB_PLATEAU_PATIENCE" --validation-fraction "$AB_VAL_FRACTION" \
		--num-workers "$AB_NUM_WORKERS" --lr "$AB_LR" --seed "$SEED" \
		--report "$AB_CONFIGS"
fi

if runs_stage ab-train; then
	echo "[ab-train] one job per arm"
	train_arms "$AB_CONFIGS" ab
	echo "Watch: squeue -u \$USER ; then: bash $0 --stages ab-eval"
fi

if runs_stage ab-eval && [[ "${EVAL_INLINE:-0}" != "1" ]]; then
	echo "[ab-eval] submitting the scoring pass"
	gpu_sbatch ab_eval "04:00:00" "ab_eval_${STAMP}" <<EOF
EVAL_INLINE=1 AB="$AB" TEST_SLP="$TEST_SLP" SCRIPTS="$SCRIPTS" \\
	bash "$SCRIPTS/$(basename "$0")" --stages ab-eval
EOF
	(( DRY_RUN )) || exit 0
fi

if runs_stage ab-eval && [[ "${EVAL_INLINE:-0}" == "1" ]]; then
	echo "[ab-eval] scoring every arm on reviewed $HOLDOUT_CAMERA"
	score_arms "$AB_CONFIGS" "$TEST_SLP" "$AB/models" "$AB/eval"
	echo
	eval_summary "$AB/eval" "$AB/arms/arms.json" | tee "$LOGS/ab_summary_${STAMP}.log"
	echo "If no strategy beats random, ranking buys nothing -- spend the budget on volume."
	echo "If none approaches oracle, no ranking rule will help much either."
fi
