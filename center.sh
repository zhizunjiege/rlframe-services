#!/bin/bash

function grace_exit {
    # kill envoy, gunicorn, python process
    echo "graceful exiting..."
    for process in envoy gunicorn python
    do
        echo "kill $process"
        kill -TERM $(ps -ef | grep $process | grep -v grep | awk '{print $2}')
    done
    echo "graceful exited"
    exit 0
}

trap grace_exit INT TERM

# run envoy in background
echo "starting envoy..."
func-e run -c envoy.yaml -l error &

# run gunicorn in background
echo "starting gunicorn..."
gunicorn web:app -w 4 -b 0.0.0.0:8080 -D

# run python in background
echo "starting python..."
python bff.py -p 10000 -w 10 -m 256 &

# wait for subprocess
wait
