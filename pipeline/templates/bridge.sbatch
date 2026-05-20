#!/bin/bash -l
#SBATCH -t 0-1
#SBATCH -c 1
#SBATCH --partition=short
#SBATCH --mem=2G
#SBATCH -J bridge
#SBATCH -o __JOBS_ROOT__/bridge_%j.out
#SBATCH -e __JOBS_ROOT__/bridge_%j.err
set -eo pipefail
shopt -s nullglob

JOBS_ROOT="__JOBS_ROOT__"
source "$JOBS_ROOT/pipeline.env"
source "$LIB_DIR/hosts.sh"

REMOTE_ROOT="$SAION_WORK_ROOT"
REMOTE_INPUT="$REMOTE_ROOT/input"
REMOTE_OUTPUT="$REMOTE_ROOT/output"
REMOTE_JOBS="$REMOTE_ROOT/jobs"
EXPORT_ROOT="/work/ReiterU/$USER/sleap_export"

# Identify model set + GPU partition so engines for different GPU types coexist.
# TRT engines are SM-specific (e.g. A100 SM80 vs V100 SM70); building on partition
# X and predicting on partition Y will fail to load the engine.
MODEL_ID="$(basename "$SLEAP_MODEL_CENTROID")__$(basename "$SLEAP_MODEL_INSTANCE")__${SAION_PARTITION}"
MODEL_ID="${MODEL_ID//\//_}"
EXPORT_DIR="$EXPORT_ROOT/$MODEL_ID"

echo "[INFO] preparing saion-side dirs"
ssh_retry saion "
	mkdir -p '$REMOTE_INPUT' '$REMOTE_OUTPUT' '$REMOTE_JOBS' '$EXPORT_ROOT' &&
	chmod 2775 '$REMOTE_INPUT' '$REMOTE_OUTPUT' '$REMOTE_JOBS'
"

# --- Lazy TRT export ----------------------------------------------------------
if (( SKIP_TRT_EXPORT == 0 )) && [[ "$SLEAP_RUNTIME" != "pytorch" ]]; then
	if ssh_retry saion "[[ -f '$EXPORT_DIR/model.trt' || -f '$EXPORT_DIR/model.onnx' ]]"; then
		echo "[OK] export already at $EXPORT_DIR"
	else
		echo "[INFO] export missing; running $SCRIPTS_DIR/export_sleap_trt.sh on saion (home is shared with deigo)"
		# export_sleap_trt.sh srun's itself onto $SAION_PARTITION and blocks until done
		ssh_retry saion "bash '$SCRIPTS_DIR/export_sleap_trt.sh' \
			--centroid '$SLEAP_MODEL_CENTROID' \
			--instance '$SLEAP_MODEL_INSTANCE' \
			--out '$EXPORT_DIR' \
			--runtime '$SLEAP_RUNTIME' \
			--partition '$SAION_PARTITION'" \
			|| { echo "[ERR] TRT export failed" >&2; exit 3; }
	fi
fi

# --- Skip rsync: saion compute self-fetches via /deigo_flash cross-mount ---
# Each saion sleap predict task does `cp /deigo_flash/.../<chunk>.mkv /work/.../input/`
# at the start of its iteration. This eliminates the deigo->saion bulk transfer
# (~hours for ~1 TB) and lets sleap start immediately after bridge exits.
WORKLIST="$JOBS_ROOT/aruco_worklist.txt"
DEIGO_FLASH_SAION_PREFIX="/deigo_flash/ReiterU/$USER/$EXP_NAME"
echo "[INFO] saion will self-fetch chunks from $DEIGO_FLASH_SAION_PREFIX (cross-mount; no rsync needed)"

# --- Render saion-side templates locally, then upload ------------------------
RENDERED_PREDICT="$JOBS_ROOT/sleap_predict_array.rendered.sh"
RENDERED_DATACP="$JOBS_ROOT/sleap_datacp_array.rendered.sh"

SLEAP_MODULE="${SLEAP_MODULE:-sleap-nn/0.2.0}"
SAION_DATACP_PARTITION="${SAION_DATACP_PARTITION:-test-gpu}"

sed -e "s#__REMOTE_JOBS__#$REMOTE_JOBS#g" \
    -e "s#__REMOTE_INPUT__#$REMOTE_INPUT#g" \
    -e "s#__REMOTE_OUTPUT__#$REMOTE_OUTPUT#g" \
    -e "s#__EXPORT_DIR__#$EXPORT_DIR#g" \
    -e "s#__SLEAP_RUNTIME__#$SLEAP_RUNTIME#g" \
    -e "s#__CHUNK_EXT__#$CHUNK_EXT#g" \
    -e "s#__SAION_PARTITION__#$SAION_PARTITION#g" \
    -e "s#__SLEAP_MODEL_CENTROID__#$SLEAP_MODEL_CENTROID#g" \
    -e "s#__SLEAP_MODEL_INSTANCE__#$SLEAP_MODEL_INSTANCE#g" \
    -e "s#__SKIP_TRT_EXPORT__#$SKIP_TRT_EXPORT#g" \
    -e "s#__SLEAP_MODULE__#$SLEAP_MODULE#g" \
    -e "s#__BATCH_SIZE__#${BATCH_SIZE:-1}#g" \
    -e "s#__DEIGO_FLASH_SAION_PREFIX__#$DEIGO_FLASH_SAION_PREFIX#g" \
    -e "s#__DATA_DIR__#$DATA_DIR#g" \
    -e "s#__HOSTS_LIB__#$LIB_DIR/hosts.sh#g" \
    "$TEMPLATES_DIR/sleap_predict_array.template.sh" > "$RENDERED_PREDICT"

sed -e "s#__REMOTE_JOBS__#$REMOTE_JOBS#g" \
    -e "s#__REMOTE_OUTPUT__#$REMOTE_OUTPUT#g" \
    -e "s#__DATA_DIR__#$DATA_DIR#g" \
    -e "s#__SAION_DATACP_PARTITION__#$SAION_DATACP_PARTITION#g" \
    -e "s#__BATCH_SIZE__#${BATCH_SIZE:-1}#g" \
    "$TEMPLATES_DIR/sleap_datacp_array.template.sh" > "$RENDERED_DATACP"

echo "[INFO] uploading worklist and rendered saion scripts"
rsync_retry -ah --chmod=Du=rwx,Dg=rwx,Fu=rwx,Fg=rwx \
	"$WORKLIST" "$RENDERED_PREDICT" "$RENDERED_DATACP" \
	"saion:$REMOTE_JOBS/"

ssh_retry saion "
	cd '$REMOTE_JOBS' &&
	mv -f sleap_predict_array.rendered.sh sleap_predict_array.sh &&
	mv -f sleap_datacp_array.rendered.sh sleap_datacp_array.sh &&
	chmod +x sleap_predict_array.sh sleap_datacp_array.sh
"

# --- Submit saion arrays ------------------------------------------------------
N=$(wc -l < "$WORKLIST")
BS="${BATCH_SIZE:-1}"
ARRAY_COUNT=$(( (N + BS - 1) / BS ))
UPPER=$(( ARRAY_COUNT - 1 ))

echo "[INFO] sbatch on saion: sleap_predict_array 0-$UPPER%$SLEAP_CONCURRENCY"
JID_SLEAP=$(ssh_retry saion "sbatch --parsable --array=0-${UPPER}%${SLEAP_CONCURRENCY} '$REMOTE_JOBS/sleap_predict_array.sh'")
echo "  saion sleap         $JID_SLEAP"
echo "$JID_SLEAP" > "$JOBS_ROOT/jid_saion_sleap.txt"

JID_SLEAP_DC=$(ssh_retry saion "sbatch --parsable --dependency=afterok:$JID_SLEAP '$REMOTE_JOBS/sleap_datacp_array.sh'")
echo "  saion sleap_datacp  $JID_SLEAP_DC"
echo "$JID_SLEAP_DC" > "$JOBS_ROOT/jid_saion_sleap_datacp.txt"

# --- Saion-side cleanup ------------------------------------------------------
ssh_retry saion "cat > '$REMOTE_JOBS/saion_cleanup.sh'" <<CLEAN
#!/bin/bash -l
#SBATCH -t 0-1
#SBATCH -c 1
#SBATCH --partition=$SAION_DATACP_PARTITION
#SBATCH --mem=4G
#SBATCH -J sleap_clean
#SBATCH -o $REMOTE_JOBS/saion_cleanup_%j.out
#SBATCH -e $REMOTE_JOBS/saion_cleanup_%j.err
set -eo pipefail
echo "[INFO] removing $REMOTE_ROOT"
rm -rf "$REMOTE_ROOT"
CLEAN
ssh_retry saion "chmod +x '$REMOTE_JOBS/saion_cleanup.sh' && sbatch --dependency=afterany:$JID_SLEAP_DC '$REMOTE_JOBS/saion_cleanup.sh'"

echo "[INFO] bridge complete"
