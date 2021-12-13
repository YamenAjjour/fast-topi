#!/bin/bash
. .deploy-config
build_docker()
{
  echo "building docker "
  docker build -t "$IMAGE_NAME" -f docker/Dockerfile .
}

run_docker()
{
  echo "runing docker "
  echo "$FAST_API_PORT"
  docker run -dit --rm -v "$(pwd)":/src -p 80:80 -w /src --name "$CONTAINER_NAME" --tty "$IMAGE_NAME"
}

run_service()
{
  echo "running service"
  docker exec -it "$CONTAINER_NAME" uvicorn api:app --host 0.0.0.0 --port 80
}

push_docker()
{
  docker login
}

build_docker
run_docker
run_service
