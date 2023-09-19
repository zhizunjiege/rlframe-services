#!/bin/bash

# parse arguments -w, -m, -l
while getopts "w:m:l:" opt; do
  case $opt in
  w)
    workers=$OPTARG
    ;;
  m)
    msglen=$OPTARG
    ;;
  l)
    loglvl=$OPTARG
    ;;
  ?)
    echo "Invalid option: -$opt"
    ;;
  esac
done

function grace_exit() {
  echo "graceful exiting..."
  for process in "$@"; do
    echo "kill $process"
    kill -TERM $(ps -ef | grep $process | grep -v grep | awk '{print $2}')
  done
  echo "graceful exited"
  exit 0
}

trap "grace_exit envoy waitress-serve python" INT TERM

# mkdir for logs
mkdir -p data/logs

# get current time
time=$(date +"%Y-%m-%d %H-%M-%S")

# run envoy in background
echo "starting envoy..."
func-e run -c envoy.yaml </dev/null >/dev/null 2>&1 &

# run waitress in background
echo "starting waitress..."
waitress-serve --listen 0.0.0.0:8888 web:app </dev/null >"data/logs/$time.web.log" 2>&1 &

# run python in background
echo "starting python..."
python -u bff.py -p 10000 -w ${workers:-10} -m ${msglen:-256} -l ${loglvl:-info}  </dev/null >"data/logs/$time.bff.log" 2>&1 &

# wait for subprocess
wait
