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

trap "grace_exit envoy gunicorn python" INT TERM

# mkdir for logs
mkdir -p data/logs

# get current time
time=$(date +"%Y-%m-%d %H-%M-%S")

# run envoy in background
echo "starting envoy..."
func-e run -c envoy.yaml -l error </dev/null >/dev/null 2>&1 &

# run gunicorn in background
echo "starting gunicorn..."
gunicorn -b 0.0.0.0:8888 --log-level info --log-file "data/logs/$time.web.log" -D web:app

# run python in background
echo "starting python..."
python -u bff.py -p 10000 -w ${workers:-10} -m ${msglen:-256} </dev/null >"data/logs/$time.bff.log" 2>&1 &

# wait for subprocess
wait
