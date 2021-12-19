import tensorflow as tf

from voxelgan.loss import Loss
from voxelgan.optimizer import Optimizer


class Latent(tf.keras.Model):
	'''
	The latent
	'''
	def __init__(self, d) -> None:
		super(Latent, self).__init__(name='')
		self.input = tf.keras.layers.InputLayer(input_shape=(d,)) #flat latent?
		self.normalize = tf.keras.layers.LayerNormalization()

	def call(self, x):
		x = self.input
		x = self.normalize(x)
		return x

class Mapping(tf.keras.Model):
	'''
	The Mapping block 
	'''
	def __init__(self, z_dim, w_dim, num_mapping_layers, lr, ) -> None:
		super(Mapping, self).__init__(name='')
		self.dense = tf.keras.layers.Dense(z_dim)
		self.leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.2)
		self.normalize = tf.keras.layers.LayerNormalization()

	def call(self, x):
		for i in range(self.num_mapping_layers):
			x = self.dense(x)
			x = self.leaky_relu(x)
		return x

class Generator_Block(tf.keras.Model):
	'''
	The Generator block 
	'''
	def __init__(self, resolution, w_dim, num_layers, lr, ) -> None:
		super(Generator_Block, self).__init__(name='')
		self.conv = tf.keras.layers.Conv3DTranspose(filters=64, kernel_size=(3,3,3), strides=(1,1,1), padding='same')
		self.leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.3)
		self.upsample = tf.keras.layers.UpSampling3D(size=(2, 2, 2))
		self.batch_norm = tf.keras.layers.BatchNormalization()
		self.affine = tf.keras.layers.Add() #TODO: Fix affine layer

	def call(self, x, m):
		x = self.affine([x, m])
		x = self.upsample(x)
		x = self.batch_norm(x)
		x = self.conv(x)
		x = self.leaky_relu(x)
		return x

class Generator(tf.keras.Model):
	def __init__(self, z_dim, resolution, w_dim, num_layers, lr, ) -> None:
		super(Generator, self).__init__()
		self.num_layers = num_layers
		self.latent = Latent(z_dim, w_dim, num_layers, lr)
		self.mapping = Mapping(z_dim, w_dim, num_layers, lr)
		self.generator_block = Generator_Block(resolution, w_dim, num_layers, lr)
		self.noise = tf.keras.layers.GaussianNoise(0.1)
		self.mapping_injector = tf.keras.layers.Add()
		self.rgbs = tf.keras.layers.Conv3D(filters=3, kernel_size=(3,3,3), strides=(1,1,1), padding='same')

	def call(self, x, training=False):
		x = self.latent(x)
		x = self.mapping(x)
		for i in range(self.num_layers): #TODO: Fix this
			x = self.generator_block([x])
			x = self.noise(x)
			x = self.mapping_injector([x, w])
		x = self.rgbs(x)
		return x


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
		#conv layer
		x = self.conv(x)
		x = self.leaky_relu(x)
		x = self.downsample(x)
		x = self.batch_norm(x)
		return x

class Discriminator(tf.keras.Model):
	def __init__(self, resolution, sequence, depth, w_dim, num_layers, filters, lr) -> None:
		super(Discriminator, self).__init__(name='')
		self.num_layers = num_layers
		self.input = tf.keras.layers.InputLayer(input_shape=(sequence, resolution, resolution, 3))
		self.conv1 = tf.keras.layers.Conv3D(filters=filters, kernel_size=(3,3,3), strides=(1,1,1), padding='same')
		self.discriminator_block = Discriminator_Block(depth, w_dim, num_layers, lr)
		self.downsample = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2))
		self.pool = tf.keras.layers.GlobalAveragePooling3D()
		self.output = tf.keras.layers.Dense(1, activation='sigmoid')

	def call(self, x, training=False):
		x = self.input(x)
		for i in range(self.num_layers): #TODO: Fix this
			x = self.discriminator_block(x)
		x = self.pool(x)
		x = self.output(x)
		return x


class GAN(tf.keras.Model):
	def __init__(self,
				generator,
				discriminator,
				latent,
				generator_metrics,
				discriminator_metrics,
				**kwargs):
		super().__init__(**kwargs)
		self.generator = generator
		self.discriminator = discriminator
		self.generator_optimizer = Optimizer(0.0002, 0.0, 0.999)
		self.discriminator_optimizer = Optimizer(0.002, 0.0, 0.999)
		self.loss = Loss()
		self.generator_metrics = generator_metrics
		self.discriminator_metrics = discriminator_metrics
		
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

			gen_loss = self.generator_loss(fake_output)
			disc_loss = self.discriminator_loss(real_output, fake_output)

		gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
		gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

		self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
		self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))


	@tf.function
	def validation_step(self, images):
		pass