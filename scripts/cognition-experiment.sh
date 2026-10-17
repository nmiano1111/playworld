#!/bin/sh
# Paired before/after experiment for world cognition.
#
# Creates PAIRS fresh worlds per cognition mode — legacy (the original
# observation and prompt) and grounded — from the same configs/world.yaml,
# roster, model, seed and LLM-call budget, runs each for DAYS simulated days,
# then prints `playworld world compare` over all of them. Modes are
# interleaved (legacy-1, grounded-1, legacy-2, ...) so drift in the machine
# or the model server affects both alike.
#
#   scripts/cognition-experiment.sh                       # 2 pairs, 1 day, 50 calls each
#   PAIRS=3 DAYS=2 CALLS=100 scripts/cognition-experiment.sh
#   MODEL=qwen3:4b scripts/cognition-experiment.sh        # a faster smoke run
#   PROVIDER=fake MODEL=fake ROOT=/tmp/x DAYS=0.25 scripts/cognition-experiment.sh   # plumbing check, no model
#
# Each world is named $PREFIX-<mode>-<n> and is replaced if it exists. The
# script refuses prefixes that could touch an existing experiment such as
# "town". Expect roughly 25 s per call with qwen3.5:9b on an M2 Pro: the
# defaults take about an hour and a half.
set -eu

PAIRS=${PAIRS:-2}
DAYS=${DAYS:-1}
CALLS=${CALLS:-50}
PREFIX=${PREFIX:-cogexp}
MODEL=${MODEL:-}
PROVIDER=${PROVIDER:-}
ROOT=${ROOT:-.playworld/worlds}

case "$PREFIX" in
  ""|town|town-*) echo "refusing PREFIX=$PREFIX: pick a name that is not an existing experiment" >&2; exit 2 ;;
esac

cd "$(dirname "$0")/.."
bin=$(mktemp -d)/playworld
go build -o "$bin" ./cmd/playworld

extra="-root $ROOT"
if [ -n "$MODEL" ]; then extra="$extra -model $MODEL"; fi
if [ -n "$PROVIDER" ]; then extra="$extra -provider $PROVIDER -skip-preflight"; fi

worlds=""
i=1
while [ "$i" -le "$PAIRS" ]; do
  for mode in legacy grounded; do
    name="$PREFIX-$mode-$i"
    echo "── $name ($(date +%H:%M)) ──"
    # shellcheck disable=SC2086
    "$bin" world run "$name" -config configs/world.yaml -fresh -cognition "$mode" \
      -days "$DAYS" -max-calls "$CALLS" -introspect -log quiet $extra
    worlds="$worlds $name"
  done
  i=$((i + 1))
done

echo
# shellcheck disable=SC2086
"$bin" world compare -root "$ROOT" $worlds
