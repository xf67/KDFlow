#!/bin/bash
# Two lanes like the four-way sweep: lane0 = GPU1 / CPUs 0-23, lane1 = GPU2 / CPUs 24-47.
# MAX_SAMPLES sized for >= 2 warm-up steps + >= 49,152 measured tokens (~850 tok/sample) with margin.
E=/home/xxf/Distill/ai_gen/kdflow-mopd-a6000-20261118; cd $E
log(){ echo "$(date +%F_%T) $*" >> $E/sweep.log; }
lane(){ local GPU=$1 CPUS=$2; shift 2
  for spec in "$@"; do BS=${spec%:*}; N=${spec#*:}; RUN=kdflow-bs$BS
    log "lane GPU$GPU start $RUN max_samples=$N"
    GPU=$GPU CPUS=$CPUS BS=$BS MAX_SAMPLES=$N RUN=$RUN harness/launch.sh
    log "lane GPU$GPU done $RUN $(cat runs/$RUN/exit_code)"
  done; }
lane 1 0-23 1:96 16:128 &
lane 2 24-47 2:96 4:104 8:112 &
wait; log "sweep finished"
