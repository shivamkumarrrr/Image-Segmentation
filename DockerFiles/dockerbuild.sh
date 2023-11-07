docker build . --build-arg USER_UID=$(id -u $(id -un)) --build-arg USER_NAME=$(id -un) -t image-segmentation
