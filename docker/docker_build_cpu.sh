#!/usr/bin/env bash

cd ..
docker build --rm=true -t olfsegnet:cpu -f ./docker/Dockerfile_CPU .