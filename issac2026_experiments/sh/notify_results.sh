#!/usr/bin/env bash
# Push each finished decoder-only run to the shared ntfy topic.
#
#   bash sh/notify_results.sh &
#
# The results live on one machine, but whoever picks the work up next may be on
# another. ntfy is the one channel that crosses: a public topic anyone can watch
# in a browser or the phone app, no account, no filesystem access.
#
#   https://ntfy.sh/calt-bench-qd-9f3a2
#
# Polls the run logs, announces every run the moment it prints its success rate,
# and posts a table once all of them are in. Safe to start late — runs that have
# already finished are announced on the first pass — and safe to run twice, since
# each announcement is recorded in a state file.
set -u
cd "$(dirname "$0")/.."

NTFY="${NTFY:-https://ntfy.sh/calt-bench-qd-9f3a2}"
STATE="${STATE:-.notify_state}"
EXPECTED="${EXPECTED:-9}"
mkdir -p "$STATE"

push() {
    curl -s -m 30 -H "Title: $1" -d "$2" "$NTFY" > /dev/null || true
}

# Every log the decoder-only runs write, across tasks.
logs() {
    ls arithmetic_addition/logs/decoder_*.log \
       arithmetic_factorization/logs/decoder_*.log \
       digit_product/logs/decoder_*.log \
       relu_recurrence/logs/decoder_*.log 2>/dev/null
}

while :; do
    done_count=0
    for log in $(logs); do
        rate=$(grep -o "Success rate: [0-9.]*%" "$log" | tail -1)
        [ -z "$rate" ] && continue
        done_count=$((done_count + 1))

        key="$STATE/$(echo "$log" | tr '/' '_')"
        [ -f "$key" ] && continue

        task=$(dirname "$(dirname "$log")")
        name=$(basename "$log" .log | sed 's/^decoder_//')
        echo "$rate" > "$key"
        push "CALT decoder-only: $task / $name" "$rate"
    done

    if [ "$done_count" -ge "$EXPECTED" ]; then
        summary=$(for log in $(logs); do
            printf '%s %s | %s\n' \
                "$(dirname "$(dirname "$log")")" \
                "$(basename "$log" .log | sed 's/^decoder_//')" \
                "$(grep -o 'Success rate: [0-9.]*%' "$log" | tail -1)"
        done)
        push "CALT decoder-only: all runs finished" "$summary"
        echo "$summary"
        break
    fi

    sleep 120
done
