@echo off

docker build -f agent.dockerfile -t rlframe-agent .

docker build -f simenv.dockerfile -t rlframe-simenv .

docker build -f center.dockerfile -t rlframe-center .
