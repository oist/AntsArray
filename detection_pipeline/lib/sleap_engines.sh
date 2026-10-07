# Per-chunk engine choice for the saion SLEAP array.
# source from sleap_predict_array.template.sh: `source "__ENGINES_LIB__"`
#
# The bridge assigns every chunk an instance cap (scripts/sleap_caps.py: nest cameras get
# the large cap, the rest the small one), writes them to sleap_caps.tsv
# (vname<TAB>chunk<TAB>cap) and renders one "cap:batch:export_dir" entry per engine it built.

# engine_for_chunk CAPS_TSV "cap:batch:dir ..." VNAME CHUNK
# Prints "cap batch dir" for the chunk. Returns 1, printing nothing, when the chunk has no
# cap row or no engine was built for its cap: guessing an engine would silently truncate
# a nest camera again, so the caller skips the chunk and it stays visibly missing.
engine_for_chunk() {
	local caps="$1" engines="$2" vname="$3" chunk="$4" cap e k b d
	cap=$(awk -F'\t' -v v="$vname" -v c="$chunk" '$1 == v && $2 == c { print $3; exit }' "$caps" 2>/dev/null)
	[[ -n "$cap" ]] || return 1
	for e in $engines; do
		IFS=: read -r k b d <<< "$e"
		if [[ "$k" == "$cap" && -n "$b" && -n "$d" ]]; then
			printf '%s %s %s\n' "$k" "$b" "$d"
			return 0
		fi
	done
	return 1
}
