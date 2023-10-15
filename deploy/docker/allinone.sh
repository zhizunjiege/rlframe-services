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

trap "grace_exit python tensorboard waitress-serve envoy" INT TERM

# mkdir for logs
mkdir -p data/logs

# get current time
time=$(date +"%Y-%m-%d %H-%M-%S")

# run bff in background
echo "starting bff..."
python -u bff.py -p 10000 -w ${workers:-10} -m ${msglen:-256} -l ${loglvl:-info} </dev/null >"data/logs/$time.bff.log" 2>&1 &

# run simenv in background
echo "starting simenv..."
python -u simenv.py -p 10001 -w ${workers:-10} -m ${msglen:-4} -l ${loglvl:-info} </dev/null >"data/logs/$time.simenv.log" 2>&1 &

# run agent in background
echo "starting agent..."
python -u agent.py -p 10002 -w ${workers:-10} -m ${msglen:-256} -l ${loglvl:-info} </dev/null >"data/logs/$time.agent.log" 2>&1 &

# run tensorboard in background
echo "starting tensorboard..."
tensorboard --host 0.0.0.0 --port 6006 --logdir=data/logs </dev/null >/dev/null 2>&1 &

# run waitress in background
echo "starting waitress..."
waitress-serve --listen 0.0.0.0:8888 web:app </dev/null >"data/logs/$time.web.log" 2>&1 &

# run envoy in background
echo "starting envoy..."
func-e run -c envoy.yaml </dev/null >/dev/null 2>&1 &

# wait for subprocess
wait
