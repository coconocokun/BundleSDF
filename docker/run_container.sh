docker rm -f bundlesdf
DIR=$(pwd)/../demo_data
xhost +  && docker run --gpus '"device=0"' --env NVIDIA_DISABLE_REQUIRE=1 -it --network=host --name bundlesdf  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v /tmp/.X11-unix:/tmp/.X11-unix -v $DIR:/home/BundleSDF/demo_data --ipc=host -e DISPLAY=${DISPLAY} -e GIT_INDEX_FILE cocoforuiuc/bundlesdf:latest bash
