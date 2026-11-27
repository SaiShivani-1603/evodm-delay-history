#!/usr/bin/env bash
# usage: scripts/run_all.sh [config_dir] [parallel_jobs]
# Runs every config in the directory, 4 at a time by default. Finished runs are skipped,
# so the script can be stopped and restarted. One log per run in logs/.
dir=${1:-configs/main}
jobs=${2:-4}
mkdir -p logs
# seed order, so an interrupted batch still has every condition at fewer seeds
ls "$dir"/*.json | sed 's/.*_s\([0-9]*\)\.json$/\1 &/' | sort -n | cut -d" " -f2 | xargs -P "$jobs" -I{} sh -c 'python run_experiment.py {} > logs/$(basename {} .json).log 2>&1'
echo ALL DONE
