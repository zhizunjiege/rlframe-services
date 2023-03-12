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

trap "grace_exit python" INT TERM

# mkdir for logs
mkdir -p data/logs

# get current time
time=$(TZ=UTC-8 date +"%Y-%m-%d %H-%M-%S")

# run python in background
echo "starting python..."
python -u simenv.py -p 10001 -w ${workers:-10} -m ${msglen:-4} </dev/null >"data/logs/$time.simenv.log" 2>&1 &

# wait for subprocess
wait
