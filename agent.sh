#!/bin/bash

# parse arguments -w, -m
while getopts "w:m:" opt; do
  case $opt in
  w)
    workers=$OPTARG
    ;;
  m)
    msglen=$OPTARG
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

trap "grace_exit python tensorboard" INT TERM

# mkdir for logs
mkdir -p data/logs

# get current time
time=$(date +"%Y-%m-%d %H-%M-%S")

# run python in background
echo "starting python..."
python -u agent.py -p 10002 -w ${workers:-10} -m ${msglen:-256} </dev/null >"data/logs/$time.agent.log" 2>&1 &

# run tensorboard in background
echo "starting tensorboard..."
tensorboard --host 0.0.0.0 --port 6006 --logdir=data/logs </dev/null >/dev/null 2>&1 &

# wait for subprocess
wait
