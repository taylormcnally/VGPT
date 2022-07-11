import os.path
import PIL
from rich.progress import (
    BarColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
)
import tensorflow as tf
import sys
import subprocess
import signal
from voxelgan import cli
import glob
from rich.progress import Progress, TaskID
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from PIL import Image
import numpy as np
import math

import torch
from megatron import print_rank_0, get_args, mpu

AUTOTUNE = tf.data.AUTOTUNE

done_event = Event()

def handle_sigint(signum, frame):
    done_event.set()


signal.signal(signal.SIGINT, handle_sigint)

progress = Progress(
    TextColumn("[bold blue] Thread {task.id}", justify="right"),
    BarColumn(bar_width=None),
    "[progress.percentage]{task.percentage:>3.1f}%",
    "•",
    TimeRemainingColumn(),
)

def _resize_image_thread(task_id: TaskID,  image_set, resolution: int):
    # Resize images in chunk
    num_images = len(image_set)
    progress.console.print(f'Resizing {num_images} images')
    progress.start_task(task_id)
    for image in image_set:
        im = Image.open(image)        
        w, h = im.size
        if w > h: # width is larger then necessary
            x, y = (w - h) // 2, 0
            im = im.crop((x, y, w - x, h))  #square crop image
        if h > resolution:
            im = im.resize((resolution, resolution), Image.ANTIALIAS) #resize if necessary
        if w > h or h > resolution:
            im.save(image)
        progress.update(task_id, advance=1)
        if done_event.is_set():
            return
    progress.console.log(f'Thread complete')



class VideoDataset(object):
    def __init__(self, data_dir: str, resolution: int, batch_size: int, sequence: int, fps: float, 
        dataproc: bool, augment: bool, workers: int):

        self.data_dir = data_dir
        self.resolution = resolution #resolution of images
        self.batch_size = batch_size
        self.sequence = sequence #the number of frames
        self.fps = fps
        self.data_proc = dataproc
        self.augment = augment #augment data
        self.workers = workers #number of threads to use for processing images
        self.dataset = None

        self.resize_and_rescale = tf.keras.Sequential([
            tf.keras.layers.Resizing(self.resolution, self.resolution),
            tf.keras.layers.Rescaling(1./127.5, offset=-1)]
        )

        self.data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(0.2),
        ])

    def prepare_data(self):
        # Check if data exists
        if not os.path.exists(self.data_dir):
            cli.print_error('Data directory not found. Please ensure the data directory is correct.')
            sys.exit(1)
        else:
            cli.print_success('Data directory found.')
            # check if data is in correct format (if its a video process it)
            if not glob.glob(self.data_dir + '/*.mp4') and not glob.glob(self.data_dir + '/*.png'):
                cli.print_error('No videos or images found in data directory. Please ensure the data directory is correct.')
                sys.exit(1)
            else:
                if not glob.glob(self.data_dir + '/*.png'):
                    self._get_images_from_videos(self.fps)
                    if self.data_proc: #process images
                        self._process_images()
            cli.print_success('Data directory is in correct format.')
            self._process_images()
            cli.print_success('Dataset prepared, Loading into memory...')

    def build_data_loader(self, dataset, drop_last=True, shuffle=False):
        """Data loader. Note that batch-size is the local (per GPU) batch-size."""
        # Sampler.
        args = get_args()
        micro_batch_size = 16
        num_workers = args.num_workers
        world_size = mpu.get_data_parallel_world_size()
        rank = mpu.get_data_parallel_rank()
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank,
            drop_last=drop_last, shuffle=shuffle
        )

        # Data loader. Note that batch size is the per GPU batch size.
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=micro_batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=num_workers,
            drop_last=not drop_last,
            pin_memory=True,
        )
        return data_loader

    def _generator(self):
        """Generate video sequences as a numpy array
        Yields:
        LENGTH x SEQUENCE x RES x RES x 3 uint8 numpy arrays
        """
        #precalc length of dataset
        files = glob.glob(self.data_dir + '*.png')
        length = int(math.ceil(len(files)/self.sequence))
        video = np.zeros(shape=(length, self.sequence, self.resolution, self.resolution, 3))
        files = sorted(files) #get all images in order by name
        for i, image_file in enumerate(files): #TODO: multithread this
            try:
                image = np.asarray(Image.open(image_file), dtype=np.uint8)
                video[i//self.sequence,i%self.sequence] = image #add image to video
            except:
                print(f'Error loading image {image_file}')
        return video
 
    def _get_images_from_videos(self, fps: float):
        #Fetch frames from mp4 videos
        task = 'getting images from videos.'
        cli.print_working(task)
        for i, video in enumerate(glob.glob(self.data_dir + '/*.mp4')):
            prefix = chr(ord('a')+i)
            subprocess.call(['ffmpeg', '-i', video, '-vf', f'fps={fps}', '-q:v', '1', '-q:a', '0', '-y', video.replace('*.mp4', f'{prefix}%d.png')])
        cli.print_done(task)

    def _process_images(self):
        #Multithreaded image processing
        images = glob.glob(self.data_dir + '/*.png')
        num_images = len(images)
        r = num_images%self.workers
        chunks = [images[i:i + int(num_images/self.workers)+r] for i in range(0, num_images, int(num_images/self.workers)+r)]
        cli.print_done(f'Divided {len(images)} into {len(chunks)} Chunks')
        with progress:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                for chunk in chunks:
                    task_id = progress.add_task('Resizing chunks', total=len(chunk), label='Resizing images')
                    pool.submit(_resize_image_thread, task_id, chunk, self.resolution)