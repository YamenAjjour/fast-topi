#!/bin/bash
. .deploy-config
build_docker()
{
  docker build -t "$IMAGE_NAME" -f docker/Dockerfile .
}

run_docker()
{
  echo "_________________________"
  echo "$FAST_API_PORT"
  #docker run -dit --rm --name "$CONTAINER_NAME" -v "$(pwd)":/src  -p "$FAST_API_PORT":"$0" "$IMAGE_NAME"
  docker run -dit  --rm --name "$CONTAINER_NAME" -p 8000:8080 -v "$(pwd)":/src  -w /src  "$IMAGE_NAME"
}

run_service()
{
  docker exec -it "$CONTAINER_NAME" uvicorn api:app
}

push_docker()
{
  docker login
}

build_docker
run_docker
run_service
