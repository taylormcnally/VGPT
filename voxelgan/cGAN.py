import tensorflow as tf

from voxelgan.loss import generator_loss, discriminator_loss
from voxelgan.optimizer import Optimizer
import numpy as np

class Latent(tf.keras.Model):
	'''
	The latent
	'''
	def __init__(self, d) -> None:
		super(Latent, self).__init__(name='')
		self.input_layer = tf.keras.layers.Input(shape=(d,))
		self.normalize = tf.keras.layers.LayerNormalization()

	def call(self, x):
		x = self.input_layer
		x = self.normalize(x)
		return x

class Mapping(tf.keras.Model):
	'''
	The Mapping block 
	'''
	def __init__(self, w_dim, m_layers) -> None:
		super(Mapping, self).__init__(name='')
		self.m_layers = m_layers
		self.dense = tf.keras.layers.Dense(w_dim)
		self.leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.2)
		self.normalize = tf.keras.layers.LayerNormalization()

	def call(self, x):
		for i in range(self.m_layers):
			x = self.dense(x)
			x = self.leaky_relu(x)
		return x

class NyaAdd(tf.keras.layers.Layer):
	#connect the mapping network with the convolutional layers
	def __init__(self) -> None:
		super(NyaAdd, self).__init__(name='NyaAdd')

	def call(self, x):
		#TODO: this was an idea (just adding the two together entirely is too many params)
		a = tf.keras.layers.Reshape(x[0].shape[-1])(x[1])
		x = tf.keras.layers.concatenate([x[0], a])
		return x

class Generator_Block(tf.keras.Model):
	'''4
	The Generator block 
	'''
	def __init__(self, sequence, filters) -> None:
		super(Generator_Block, self).__init__(name='')
		self.conv = tf.keras.layers.Conv3DTranspose(filters=filters, kernel_size=(3,3,3), strides=(1,1,1), padding='same')
		self.leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.3)
		self.upsample = tf.keras.layers.UpSampling3D(size=(2, 2, 2))
		self.batch_norm = tf.keras.layers.BatchNormalization()

	def call(self, x):
		x = self.upsample(x)
		x = self.batch_norm(x)
		x = self.conv(x)
		x = self.leaky_relu(x)
		return x

class RGB_Block(tf.keras.Model):
	'''
	The RGB block 
	'''
	def __init__(self, sequence, filters) -> None:
		super(RGB_Block, self).__init__(name='')
		self.conv = tf.keras.layers.Conv3D(filters=filters, kernel_size=(3,3,3), strides=(1,1,1), padding='same')
		self.leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.3)
		self.batch_norm = tf.keras.layers.BatchNormalization()

	def call(self, x):
		x = self.conv(x)
		x = self.leaky_relu(x)
		x = self.batch_norm(x)
		return x

class Generator(tf.keras.Model):
	def __init__(self, resolution, sequence, filters, z_dim, w_dim, mapping_layers) -> None:
		super(Generator, self).__init__()
		self.latent = Latent(w_dim)
		self.mapping = Mapping(w_dim, mapping_layers)
		self.generator_block = Generator_Block(sequence, filters)
		self.rgbs = RGB_Block(sequence, filters)
		self.nya = NyaAdd()
		init = np.zeros((1,4,4,4,512))

		self.const = tf.Variable(init, shape=(1,4,4,4,512), trainable=True)
		self.noise = tf.keras.layers.GaussianNoise(0.1)

	def call(self, x, training=False):
		x = self.latent(x)
		x = self.mapping(x)
		z = self.const
		for i in range(4): #TODO: Fix this
			z = self.generator_block(z)
			z = self.noise(z)
			z = self.nya([z,x])
		z = self.rgbs(z)
		return z


class Discriminator_Block(tf.keras.Model):
	'''
	The Discriminator block 
	'''
	def __init__(self, filters) -> None:
		super(Discriminator_Block, self).__init__(name='')
		self.conv = tf.keras.layers.Conv3D(filters=filters, kernel_size=(3,3,3), strides=(1,1,1), padding='same')
		self.leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.3)
		self.downsample = tf.keras.layers.AveragePooling3D(pool_size=(2, 2, 2))
		self.batch_norm = tf.keras.layers.BatchNormalization()

	def call(self, x):
		x = self.conv(x)
		x = self.leaky_relu(x)
		x = self.downsample(x)
		x = self.batch_norm(x)
		return x

class Discriminator(tf.keras.Model):
	def __init__(self, resolution, sequence, filters) -> None:
		super(Discriminator, self).__init__(name='')
		self.input_layer = tf.keras.layers.Input(shape=(None, sequence, resolution, resolution, 3))
		self.conv1 = tf.keras.layers.Conv3D(filters=filters, kernel_size=(3,3,3), strides=(1,1,1), padding='same')
		self.discriminator_block = Discriminator_Block(filters)
		self.downsample = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2))
		self.pool = tf.keras.layers.GlobalAveragePooling3D()
		self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid')

	def call(self, x, training=False):
		x = self.input_layer(x)
		for i in range(4): #TODO: Fix this
			x = self.discriminator_block(x)
		x = self.pool(x)
		x = self.output_layer(x)
		return x


class cGAN(tf.keras.Model):
	def __init__(self,
				generator,
				discriminator,
				generator_metrics=None,
				discriminator_metrics=None,
				generator_lr=0.0001,
				discriminator_lr=0.0001,
				**kwargs):
		super().__init__(**kwargs)
		self.generator = generator.build(input_shape=(None,512))
		self.discriminator = discriminator.build(input_shape=(None,32,512,512,3))
		self.generator_optimizer = Optimizer(generator_lr, 0.0, 0.999)
		self.discriminator_optimizer = Optimizer(discriminator_lr, 0.0, 0.999)
		self.generator_metrics = generator_metrics
		self.discriminator_metrics = discriminator_metrics
	
		self.discriminator.build()

		self.generator.build()

		#sanity checks
		# assert self.generator.output_shape == self.discriminator.input_shape



		#print the summary of the model
		self.generator.summary()
		tf.keras.utils.plot_model(self.generator, to_file='generator.png', show_shapes=True)
		self.discriminator.summary()
		tf.keras.utils.plot_model(self.discriminator, to_file='discriminator.png', show_shapes=True)


	def compile(self,
				optimizer,
				loss,
				metrics=None,
				**kwargs):
		super().compile(optimizer, loss, metrics, **kwargs)

	@tf.function
	def train_step(self, images):
		noise = tf.random.normal([BATCH_SIZE, noise_dim])

		with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
			generated_images = self.generator(noise, training=True)

			real_output = self.discriminator(images, training=True)
			fake_output = self.discriminator(generated_images, training=True)

			gen_loss = generator_loss(fake_output)
			disc_loss = discriminator_loss(real_output, fake_output)

		gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
		gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

		self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
		self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))


	@tf.function
	def validation_step(self, images):
		pass