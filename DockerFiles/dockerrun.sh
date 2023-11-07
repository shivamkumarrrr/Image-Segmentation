docker run -it --rm --runtime=nvidia --pid=host --ipc=host --user=bhpo00001 -p 7000:8888 \
-v ~/projects/image-segmentation:/image-segmentation \
-v /raid/bhpo00001/pre-trained-image-segmentation:/pre-trained-image-segmentation \
-v /raid/bhpo00001/datasets:/datasets \
-v /raid/bhpo00001/logs/image-segmentation/logfiles:/logfiles \
-v /raid/bhpo00001/logs/image-segmentation/checkpoints:/checkpoints \
-v /raid/bhpo00001/logs/image-segmentation/tb-logs:/tb-logs \
-v /raid/bhpo00001/logs/image-segmentation/wandb-logs:/wandb-logs \
image-segmentation
