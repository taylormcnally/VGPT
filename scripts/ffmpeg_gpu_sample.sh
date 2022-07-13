export out_w=512
export out_h=512

docker run --rm -it --runtime=nvidia \
    --volume $PWD:/workspace \
    willprice/nvidia-ffmpeg \
      -i $1 \
      -vf scale="$out_w:$out_h" \
      resize/$2