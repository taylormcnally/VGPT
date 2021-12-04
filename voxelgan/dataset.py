import os.path
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
from animax import cli
import glob
from rich.progress import Progress, TaskID
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from PIL import Image
import numpy as np

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


class Dataset(object):
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
            cli.print_success('Dataset prepared.')

    def load_data(self):
        #Loads data from disk into tensorflow dataset

        self.dataset = tf.data.Dataset.from_generator(self._generator, output_signature=(
                    tf.TensorSpec(shape=(self.sequence,self.resolution,self.resolution,3), dtype=tf.int16)))
        self.dataset.take(1)
        self.dataset = self.dataset.map(lambda x, y: (self.resize_and_rescale(x), y), 
              num_parallel_calls=AUTOTUNE)

        if self.augment: 
            self.dataset = self.dataset.map(lambda x, y: (self.data_augmentation(x, training=True), y), 
                num_parallel_calls=AUTOTUNE)

        self.dataset.batch(self.batch_size)
        self.dataset.cache().prefetch(buffer_size=AUTOTUNE) #use buffered prefetching on the dataset

    def _generator(self):
        """Generate video sequences.
        Yields:
        SEQUENCE x RES x RES x 3 uint8 numpy arrays
        """
        video = np.zeros(shape=(self.sequence, self.resolution, self.resolution, 3), dtype=np.uint8)
        with glob.glob(self.data_dir + '/*.png') as f: #get all images in order by name
            for image_file, i in enumerate(f):
                image = tf.io.decode_png(image_file)
                if i%self.sequence == 0 and i != 0: #if we have a full sequence
                    yield {
                        'video': video,
                    }
                    video = np.zeros(shape=(self.sequence, self.resolution, self.resolution, 3), dtype=np.uint8) #reset video
                video[i%self.sequence] = image #add image to video
 
    def _get_images_from_videos(self, fps: float):
        #Fetch frames from videos
        task = 'getting images from videos.'
        cli.print_working(task)
        for video in glob.glob(self.data_dir + '/*.mp4'):
            subprocess.call(['ffmpeg', '-i', video, '-vf', f'fps={fps}', '-q:v', '1', '-q:a', '0', '-y', video.replace('.mp4', f'%03d.png')])
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