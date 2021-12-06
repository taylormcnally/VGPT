import tensorflow as tf
import cli 


def get_distribute_strategy():
    """
    Get tensorflow distribute strategy.
    """
    try:
        #TODO: Add support for TPUs
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        strategy = tf.distribute.TPUStrategy(tpu)
    except ValueError:
        gpus = tf.config.list_logical_devices('GPU')
        strategy = tf.distribute.MirroredStrategy(gpus)
    return strategy



def get_gpu_info(mode):
    """
    Get GPU Hardware info. GPUs are preferred, warn if not available.
    """
    physical_devices = tf.config.list_physical_devices('GPU')
    if mode == 'train':
        if physical_devices != '':
            cli.print_success(f'Found GPUs at: {physical_devices}')
        else:
            cli.print_warning('No hardware accelerator detected! Are you sure you want to continue?')   
    return physical_devices





