#!/usr/bin/env bash
# Classical lex-vs-degrevlex timing, to sit next to the learning gap of Table 5.
#
#   bash sh/measure_gb_timing.sh
#
# Two parts:
#   (a) the paper's own setting, 2 variables and F=[f1,f2], over GF(7) and QQ;
#   (b) a sweep in the number of variables on square systems (n polynomials in
#       n variables, generically zero-dimensional), which is where the classical
#       lex penalty is supposed to appear.
#
# Both orders are timed with libsingular:slimgb. stdfglm is not used: it is the
# degrevlex+FGLM route, so it would measure the conversion strategy instead of a
# direct lex computation.
set -u
cd "$(dirname "$0")/.."

SAGE_BIN="${SAGE_BIN:-$HOME/micromamba/envs/sage/bin}"
CALT_SRC="${CALT_SRC:-$HOME/calt_issac_rerun/src}"
RUN="env PATH=$SAGE_BIN:$PATH PYTHONPATH=$CALT_SRC $SAGE_BIN/python measure_gb_timing.py"
OUT="gb_timing_results.json"

# (a) the paper's setting
for cfg in configs/data_GF7.yaml configs/data.yaml; do
    echo "=== paper setting: $cfg"
    $RUN --config_path "$cfg" --num_samples 1000 --output "$OUT"
done

# (b) scaling in the number of variables, square systems
for cfg in configs/data_GF7.yaml configs/data.yaml; do
    for spec in "x,y:2" "x,y,z:3" "x,y,z,w:4" "x,y,z,w,v:5"; do
        syms="${spec%%:*}"
        npoly="${spec##*:}"
        echo "=== sweep: $cfg  vars=$syms  npoly=$npoly"
        $RUN --config_path "$cfg" \
             --symbols "$syms" \
             --num_polynomials "$npoly" \
             --num_samples 200 \
             --timeout_s 120 \
             --output "$OUT"
    done
done

# (c) control: does the near-1 ratio at the paper's setting mean "no classical
# penalty here", or is a 0.07 ms computation simply too small to measure? Raising
# the degree at two variables answers it — if the ratio climbs, the measurement
# is sensitive and the flat result at degree 4 is a fact about the problem size.
for degree in 8 16 32; do
    echo "=== degree control: GF7, 2 variables, max_degree=$degree"
    $RUN --config_path configs/data_GF7.yaml \
         --max_degree "$degree" \
         --num_samples 200 \
         --timeout_s 120 \
         --output gb_timing_degree_control.json
done

echo "all timings written to $OUT and gb_timing_degree_control.json"
