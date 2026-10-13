# Crop budget of the exported top-down engine.
# source from pipeline.sh and scripts/export_sleap_trt.sh
#
# The centered-instance model runs on cap x batch square crops at once, and TensorRT cannot
# build the engine above a limit that shrinks as the crop grows ("no tactics to implement
# ... MaxPool", with any workspace size). Measured on A100:
#   640: 20x8, 48x4, 64x3, 96x2, 128x1 build; 240, 256, 288 fail (2026-10-05)
#   768: 96x1, 20x7 build; 160, 192 fail (2026-10-13)
# An unmeasured crop has no budget: a guess that is too large fails only after a long build,
# one that is too small silently costs speed. Measure it, then add a row.

# instance_crop_size MODEL_DIR
# Prints the crop_size from the model's training_config.yaml (sleap-nn writes it under
# data_config.preprocessing). Returns 1, printing nothing, when there is no config or the
# crop is not a fixed number (null = sized from the data at training time).
instance_crop_size() {
	local cfg="$1/training_config.yaml" crop
	[[ -r "$cfg" ]] || return 1
	crop=$(sed -n 's/^[[:space:]]*crop_size:[[:space:]]*\([0-9][0-9]*\)[[:space:]]*\(#.*\)\{0,1\}$/\1/p' "$cfg" | head -1)
	[[ -n "$crop" ]] || return 1
	echo "$crop"
}

# max_engine_crops CROP
# Prints the largest cap x batch that builds at this crop size; returns 1 for a crop
# that has not been measured.
max_engine_crops() {
	case "$1" in
		640) echo 192 ;;
		768) echo 140 ;;
		*) return 1 ;;
	esac
}
