import os
from warnings import filters
import click
from tensorflow.keras import mixed_precision
import tensorflow as tf
import datetime
import time

from tensorflow.python.eager.context import device
from voxelgan.GAN import GAN, Discriminator, Generator, Latent
from voxelgan.dataset import VideoDataset
from voxelgan.utils import *

title = '''
functional

'''


def parse_comma_separated_list(s):
    if isinstance(s, list):
        return s
    if s is None or s.lower() == 'none' or s == '':
        return []
    return s.split(',')


cmd_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "commands"))


# some of this is taken from stylegan3
@click.command()
# Required.
@click.option('--data',         help='Training data', metavar='[DIR]',                          type=str, default='data/')
@click.option('--res',          help='The model resolution', metavar='INT',                     type=click.IntRange(min=256, max=2048), default=512)
@click.option('--seq',          help='The sequence size', metavar='INT',                        type=click.IntRange(min=8, max=128), default=32)
@click.option('--batch',        help='Total batch size', metavar='INT',                         type=click.IntRange(min=1), default=16)
@click.option('--epochs',       help='Total number of epochs', metavar='INT',                   type=click.IntRange(min=1), default=100)
@click.option('--aug',          help='Augmentation mode',                                       type=bool, default=False)
@click.option('--filters',      help='filters',                                                 type=int, default=64)
@click.option('--map-depth',    help='Mapping network depth  [default: varies]', metavar='INT', type=click.IntRange(min=1), default=8)
@click.option('--z-dim',        help='Z dimensions', metavar='INT',                             type=click.IntRange(min=8), default=512)
@click.option('--w-dim',        help='W dimensions', metavar='INT',                             type=click.IntRange(min=8), default=512)


# Optional features.
@click.option('--freezed',      help='Freeze first layers of D', metavar='INT',                 type=click.IntRange(min=0), default=0, show_default=True)
@click.option('--data-proc',     help='process and reshape all images', metavar='INT',                 type=bool, default=True, show_default=True)

# Misc hyperparameters.
@click.option('--fps',          help='Framrate of the video extraction', metavar='FLOAT',       type=click.FloatRange(max=60), default=1.0, show_default=True)
@click.option('--batch-gpu',    help='Limit batch size per GPU', metavar='INT',                 type=click.IntRange(min=1))
@click.option('--cbase',        help='Capacity multiplier', metavar='INT',                      type=click.IntRange(min=1), default=32768, show_default=True)
@click.option('--glr',          help='G learning rate  [default: varies]', metavar='FLOAT',     type=click.FloatRange(min=0))
@click.option('--dlr',          help='D learning rate', metavar='FLOAT',                        type=click.FloatRange(min=0), default=0.002, show_default=True)
@click.option('--mbstd-group',  help='Minibatch std group size', metavar='INT',                 type=click.IntRange(min=1), default=4, show_default=True)

# Misc settings.
@click.option('--dir',          help='The checkpoint save directory', metavar='[PATH|URL]',     type=str, default='training/', show_default=True)
@click.option('--metrics',      help='Quality metrics', metavar='[NAME|A,B,C|none]',            type=parse_comma_separated_list, default='fid50k_full', show_default=True)
@click.option('--kimg',         help='Total training duration', metavar='KIMG',                 type=click.IntRange(min=1), default=25000, show_default=True)
@click.option('--snap',         help='How often to save snapshots', metavar='TICKS',            type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--seed',         help='Random seed', metavar='INT',                              type=click.IntRange(min=0), default=0, show_default=True)
@click.option('--fp32',         help='Enable mixed-precision', metavar='BOOL',                  type=bool, default=True, show_default=True)
@click.option('--workers',      help='DataLoader worker processes', metavar='INT',              type=click.IntRange(min=1), default=4, show_default=True)


def train(**kwargs):
    print(title)


    physical_devices = get_gpu_info('train')

    if physical_devices:
        strategy = get_distribute_strategy()


    #ensure that the dataset is in the correct format
    # dataset = VideoDataset(kwargs['data'], kwargs['res'], kwargs['batch'], kwargs['seq'], kwargs['fps'], kwargs['data_proc'], kwargs['aug'] ,kwargs['workers'])
    # dataset.prepare_data()
    # dataset.load_data()
    #if augmentation is enabled, add preprocessing layers to the GAN.

    #mixed precision training
    if kwargs['fp32']:
        mixed_precision.set_global_policy('mixed_float16')

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    #create the GAN
    generator = Generator(resolution=kwargs['res'], sequence=kwargs['seq'], filters=kwargs['filters'], z_dim=kwargs['z_dim'], w_dim=kwargs['w_dim'], mapping_layers=kwargs['map_depth'])
    discriminator = Discriminator(resolution=kwargs['res'], sequence=kwargs['seq'], filters=kwargs['filters'])

    gan = GAN(generator, discriminator, generator_lr=kwargs['glr'], discriminator_lr=kwargs['dlr'])

    checkpoint = tf.train.Checkpoint(generator=gan.generator,
                                    discriminator=gan.discriminator,
                                    generator_optimizer=gan.generator_optimizer,
                                    discriminator_optimizer=gan.discriminator_optimizer)

    manager = tf.train.CheckpointManager(checkpoint, kwargs['dir'], max_to_keep=3)

    # if a checkpoint exists, restore the latest checkpoint.
    if manager.latest_checkpoint and kwargs['resume'] == True:
        checkpoint.restore(manager.latest_checkpoint)
        print ('Latest checkpoint restored!!')


    train_summary_writer = tf.summary.create_file_writer(kwargs['dir'])


    for epoch in range(kwargs['epochs']):
        start_time = time.time()
        for image_batch in dataset:
            gan.train_step(image_batch)
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)


        # Run a validation loop at the end of each snapshot and save
        if epoch % kwargs['snap'] == 0:
            print ('Saving checkpoint for epoch {} at {}'.format(epoch+1, kwargs['dir']))
            gan.validation_step(epoch)
            checkpoint.save(kwargs['dir'] + 'GAN_epoch_{}'.format(epoch))


        # print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, time.time()-start_time))
        # template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'

        # print(template.format(epoch+1,
        #                  train_loss.result(), 
        #                  train_accuracy.result()*100)

if __name__ == '__main__':
    train()

