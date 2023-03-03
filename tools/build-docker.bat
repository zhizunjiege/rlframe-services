@echo off

docker build -f agent.dockerfile -t rlframe-agent .

docker build -f simenv.dockerfile -t rlframe-simenv .

docker build -f bff.dockerfile -t rlframe-bff .

docker build -f web.dockerfile -t rlframe-web .
