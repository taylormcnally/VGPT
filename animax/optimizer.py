import tensorflow as tf


def Optimizer(lr, beta1, beta2):
		return tf.keras.optimizers.Adam(
		learning_rate=lr, beta_1=beta1, beta_2=beta2, epsilon=1e-07, amsgrad=False,
		name='Adam')