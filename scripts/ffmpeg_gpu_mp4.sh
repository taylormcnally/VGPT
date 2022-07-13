docker run --rm -it --runtime=nvidia \
    --volume $PWD:/workspace \
    willprice/nvidia-ffmpeg \
      -hwaccel_device 0 \
      -hwaccel cuda \
      -hwaccel_output_format cuda \
      -i $1 \
      -c:v h264_nvenc \
      videos/$1-output.mp4
