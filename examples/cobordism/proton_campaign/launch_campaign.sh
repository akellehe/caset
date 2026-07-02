#!/usr/bin/env bash
# Throwaway campaign supervisor (#555): wait for any prior-generation workers to finish
# their in-flight attempts (never oversubscribe the 16-CPU cap), then launch 16
# animated sweep workers against the calendar-pinned deadline. Safe to re-run after a
# reboot: the deadline file persists and the workers skip already-recorded seeds.
set -u
cd /home/andrew/feat-proton-ingredients
RUN=.overnight
END_FILE=$RUN/campaign_deadline
if [ ! -f "$END_FILE" ]; then
  echo $(( $(date +%s) + 30 * 24 * 3600 )) > "$END_FILE"
fi
DEADLINE=$(cat "$END_FILE")
echo "$(date): campaign deadline $(date -d @"$DEADLINE")"

while :; do
  ALIVE=$(ps -eo args | grep -c "[w]orker.py --worker")
  [ "$ALIVE" -eq 0 ] && break
  echo "$(date): waiting for $ALIVE prior workers to finish their in-flight attempts"
  sleep 300
done

# Workers are plain children and this script WAITS on them: run under
# `systemd-run --user` the service's main process is this script, so the unit —
# and with it the workers' cgroup — stays alive for the whole campaign. (Detached
# setsid children die with the harness session; a long campaign must live in its
# own unit.)
rm -f $RUN/*.jsonl.done
for i in $(seq 0 15); do
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
  .venv-build/bin/python $RUN/worker.py --worker $i \
    --deadline "$DEADLINE" --out $RUN/worker_$i.jsonl \
    --seed-base $(( 1000000 * (i + 1) + 1000 )) --animate \
    > $RUN/worker_$i.log 2>&1 &
done
sleep 5
ps -eo pid,args | grep "[w]orker.py --worker" | awk '{print $1}' > $RUN/worker.pids
echo "$(date): campaign launched: $(wc -l < $RUN/worker.pids) workers, deadline $(date -d @"$DEADLINE")"
wait
echo "$(date): campaign ended (all workers exited)"
