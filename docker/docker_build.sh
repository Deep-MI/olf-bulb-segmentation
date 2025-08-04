#!/bin/bash
cd ..
docker build --rm=true -t olfsegnet:gpu -f ./docker/Dockerfile_nloc .