#!/bin/bash

docker build -f agent.dockerfile -t agent .

docker build -f simenv.dockerfile -t simenv .

docker build -f bff.dockerfile -t bff .

docker build -f web.dockerfile -t web .
