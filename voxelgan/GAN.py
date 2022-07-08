import tensorflow as tf

from typing import Any, Callable, List, Optional, Text, Tuple, Union
import numpy as np

_CHR_IDX = str.ascii_lowercase

def block_voxels(inputs: tf.Tensor, patch_size: int) -> tf.Tensor:
	# see https://github.com/google-research/hit-gan/blob/b7671bc9215056632dd0b4cb7701bcf644ec3ad0/models/generators.py
	"""Converts the voxel to blocked patches."""
	# inputs: (batch_size, sequence, height, width, channels)　
	# outputs: (batch_size, grid_s * grid_h * grid_w, patch_s * patch_h * patch_w, channels)
	_, sequence, height, width, channel_dim = inputs.shape
	patch_length = patch_size**3

	outputs = tf.nn.space_to_depth(inputs, patch_size)
	outputs = tf.reshape(
				outputs,
				shape=(-1, (sequence * height * width) // patch_length, patch_length, channel_dim))
	return outputs


def block_images(inputs: tf.Tensor, patch_size: int) -> tf.Tensor:
	"""Converts the image to blocked patches."""
	# inputs: (batch_size, height, width, channels)
	_, height, width, channel_dim = inputs.shape
	patch_length = patch_size**2

	outputs = tf.nn.space_to_depth(inputs, patch_size)
	outputs = tf.reshape(
		outputs,
		shape=(-1, height * width // patch_length, patch_length, channel_dim))
	# outputs: (batch_size, grid_h * grid_w, patch_h * patch_w, channels)
	return 


def unblock_voxels(inputs: tf.Tensor, grid_size: int,
				   patch_size: int) -> tf.Tensor:
	"""Converts blocked patches to voxels."""
	# inputs: (batch_size, grid_h * grid_w, patch_s * patch_h * patch_w, channels)
	# outputs: (batch_size, height, width, channels)

	grid_width = grid_size
	grid_height = inputs.shape[1] // grid_width
	grid_sequence = inputs.shape[0]
	channel_dim = inputs.shape[3]

	outputs = tf.reshape(
		inputs,
		shape=(-1, grid_sequence, grid_height, grid_width, patch_size**3 * channel_dim))
	outputs = tf.nn.depth_to_space(outputs, patch_size)
	return outputs


def _3dconvolution():
		
	
	
	pass


def _build_attention_equation(rank, attn_axes):
	"""Builds einsum equations for the attention computation.
	Query, key, value inputs after projection are expected to have the shape as:
	(bs, <non-attention dims>, <attention dims>, num_heads, channels).
	bs and <non-attention dims> are treated as <batch dims>.
	The attention operations can be generalized:
	(1) Query-key dot product:
	(<batch dims>, <query attention dims>, num_heads, channels), (<batch dims>,
	<key attention dims>, num_heads, channels) -> (<batch dims>,
	num_heads, <query attention dims>, <key attention dims>)
	(2) Combination:
	(<batch dims>, num_heads, <query attention dims>, <key attention dims>),
	(<batch dims>, <value attention dims>, num_heads, channels) -> (<batch dims>,
	<query attention dims>, num_heads, channels)
	Args:
	rank: the rank of query, key, value tensors.
	attn_axes: a list/tuple of axes, [-1, rank), that will do attention.
	Returns:
	Einsum equations.
	"""
	target_notation = _CHR_IDX[:rank]
	# `batch_dims` includes the head dim.
	batch_dims = tuple(np.delete(range(rank), attn_axes + (rank - 1,)))
	letter_offset = rank
	source_notation = ""
	for i in range(rank):
		if i in batch_dims or i == rank - 1:
			source_notation += target_notation[i]
		else:
			source_notation += _CHR_IDX[letter_offset]
			letter_offset += 1

	product_notation = "".join([target_notation[i] for i in batch_dims] +
								[target_notation[i] for i in attn_axes] +
								[source_notation[i] for i in attn_axes])
	dot_product_equation = "%s,%s->%s" % (source_notation, target_notation,
										product_notation)
	attn_scores_rank = len(product_notation)
	combine_equation = "%s,%s->%s" % (product_notation, source_notation,
									target_notation)
	return dot_product_equation, combine_equation, attn_scores_rank


def _build_proj_equation(free_dims, bound_dims, output_dims):
	"""Builds an einsum equation for projections inside multi-head attention."""
	input_str = ""
	kernel_str = ""
	output_str = ""
	bias_axes = ""
	letter_offset = 0
	for i in range(free_dims):
		char = _CHR_IDX[i + letter_offset]
		input_str += char
		output_str += char

	letter_offset += free_dims
	for i in range(bound_dims):
		char = _CHR_IDX[i + letter_offset]
		input_str += char
		kernel_str += char

	letter_offset += bound_dims
	for i in range(output_dims):
		char = _CHR_IDX[i + letter_offset]
		kernel_str += char
		output_str += char
		bias_axes += char
		equation = "%s,%s->%s" % (input_str, kernel_str, output_str)

	return equation, bias_axes, len(output_str)


def _get_output_shape(output_rank, known_last_dims):
	return [None] * (output_rank - len(known_last_dims)) + list(known_last_dims)


def make_norm_layer(
	norm_type: Optional[Text] = "batch") -> tf.keras.layers.Layer:
	"""Makes the normalization layer.
	Args:
	norm_type: A string for the type of normalization.
	Returns:
	A `tf.keras.layers.Layer` instance.
	"""
	if norm_type is None:
		return tf.keras.layers.Layer()  # Identity.
	elif norm_type == "batch":
		return tf.keras.layers.BatchNormalization()
	elif norm_type == "syncbatch":
		return tf.keras.layers.experimental.SyncBatchNormalization()
	elif norm_type == "layer":
		return tf.keras.layers.LayerNormalization(epsilon=1e-6)
	else:
		raise ValueError("{} is not a recognized norm type".format(norm_type))


class PositionEmbedding(tf.keras.layers.Layer):
	"""Defines learnable positional embeddings."""
	def build(self, input_shape: tf.TensorShape) -> None:
		input_dim = input_shape[-1]
		input_height = input_shape[-3]
		input_width = input_shape[-2]
		self.embedding_weight = self.add_weight(
			"embedding_weight",
			shape=(1, input_height, input_width, input_dim),
			initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
			trainable=True)
		super().build(input_shape)

	def call(self, inputs: tf.Tensor) -> tf.Tensor:
		return inputs + self.embedding_weight


class SkipToVoxel(tf.keras.layers.Layer):
  """Converts skip inputs to RGB images."""

	def __init__(self,
				output_dim: int = 4,
				norm_type: Text = "layer",
				kernel_initializer="glorot_uniform",
				bias_initializer="zeros",
				**kwargs: Any) -> None:
		"""Initializer.
		Args:
			output_dim: An integer for the output channel dimension.
			norm_type: A string for the type of normalization.
			kernel_initializer: Initialization function of dense kenrels.
			bias_initializer: Initialization function of dense biases.
			**kwargs: Additional arguments for `tf.keras.layers.Layer`.
		"""
		super().__init__(**kwargs)
		self.output_layer = tf.keras.Sequential([
			make_norm_layer(norm_type),
			tf.keras.layers.Dense(
				output_dim,
				kernel_initializer=kernel_initializer,
				bias_initializer=bias_initializer)
		])
		self.upsample = tf.keras.layers.UpSampling3D()

	def call(self,
			inputs: tf.Tensor,
			skip_inputs: Optional[tf.Tensor],
			training: Optional[bool] = None) -> tf.Tensor:
		outputs = self.output_layer(inputs, training=training)
		if skip_inputs is not None:
			skip_outputs = self.upsample(skip_inputs)
		outputs = skip_outputs + outputs
		return outputs


	class PixelShuffle(tf.keras.layers.Layer):
		"""Up-sampling layer using pixel shuffle."""

	def __init__(self,
				output_dim: int,
				kernel_initializer="glorot_uniform",
				bias_initializer="zeros",
				**kwargs: Any) -> None:
	"""Initializer.
	Args:
		output_dim: An integer for the output channel dimension.
		kernel_initializer: Initialization function of dense kenrels.
		bias_initializer: Initialization function of dense biases.
		**kwargs: Additional arguments for `tf.keras.layers.Layer`.
	"""
	super().__init__(**kwargs)
	self._output_dim = output_dim
	self._kernel_initializer = kernel_initializer
	self._bias_initializer = bias_initializer

	def build(self, input_shape: tf.TensorShape) -> None:
	if input_shape[-1] // 4 == self._output_dim:
	self.dense_layer = None
	else:
		self.dense_layer = tf.keras.layers.Dense(
			self._output_dim,
			kernel_initializer=self._kernel_initializer,
			bias_initializer=self._bias_initializer)
	super().build(input_shape)

	def call(self, inputs: tf.Tensor) -> tf.Tensor:
	outputs = tf.nn.depth_to_space(inputs, 2)
	if self.dense_layer is not None:
		outputs = self.dense_layer(outputs)
	return outputs


class MLP(tf.keras.layers.Layer):
  """Defines MLP layer with normalization and residual connection."""

  def __init__(self,
			   expansion: int = 4,
			   dropout: float = 0.,
			   norm_type: Text = "batch",
			   activation: Callable[..., tf.Tensor] = tf.nn.relu,
			   kernel_initializer="glorot_uniform",
			   bias_initializer="zeros",
			   **kwargs: Any) -> None:
	"""Initializer.
	Args:
	  expansion: An integer for the expansion ratio of the hidden dimension.
	  dropout: A float for the dropout rate after dense layers.
	  norm_type: A string for the type of normalization.
	  activation: Activation function.
	  kernel_initializer: Initialization function of dense kenrels.
	  bias_initializer: Initialization function of dense biases.
	  **kwargs: Additional arguments for `tf.keras.layers.Layer`.
	"""
	super().__init__(**kwargs)
	self._expansion = expansion
	self._dropout = dropout
	self._norm_type = norm_type
	self._activation = activation
	self._kernel_initializer = kernel_initializer
	self._bias_initializer = bias_initializer

  def build(self, input_shape: tf.TensorShape) -> None:
	input_dim = input_shape[-1]
	common_kwargs = dict(
		kernel_initializer=self._kernel_initializer,
		bias_initializer=self._bias_initializer)

	self.norm_layer = make_norm_layer(self._norm_type)
	self.mlp_block = tf.keras.Sequential([
		tf.keras.layers.Dense(
			input_dim * self._expansion,
			activation=self._activation,
			**common_kwargs),
		tf.keras.layers.Dropout(self._dropout),
		tf.keras.layers.Dense(input_dim, **common_kwargs),
		tf.keras.layers.Dropout(self._dropout)
	])
	super().build(input_shape)

  def call(self,
		   inputs: tf.Tensor,
		   training: Optional[bool] = None) -> tf.Tensor:
	outputs = self.norm_layer(inputs, training=training)
	outputs = self.mlp_block(outputs, training=training)
	return outputs + inputs


class MultiAxisAttention(tf.keras.layers.Layer):
  """MultiAxisAttention performs attentions along multiple axes."""

  def __init__(self,
			   num_heads: int,
			   key_dim: int,
			   attn_axes: List[List[int]],
			   attn_type: Text = "multi_head",
			   use_bias: bool = True,
			   dropout: float = 0.0,
			   kernel_initializer="glorot_uniform",
			   bias_initializer="zeros",
			   **kwargs: Any) -> None:
	"""Initializer.
	Args:
	  num_heads: An integer for the number of attention heads.
	  key_dim: An integer for the size of each attention head.
	  attn_axes: A list for the list of axes over which the attention is
		applied.
	  attn_type: A string for attention type ("multi_head" or "multi_query").
	  use_bias: A boolean for whether the dense layers use biases.
	  dropout: A float for the dropout rate after dense layers.
	  kernel_initializer: Initialization function of dense kenrels.
	  bias_initializer: Initialization function of dense biases.
	  **kwargs: Additional arguments for `tf.keras.layers.Layer`.
	"""
	super().__init__(**kwargs)
	self._num_heads = num_heads
	self._key_dim = key_dim
	self._attn_axes = attn_axes
	self._attn_type = attn_type
	self._use_bias = use_bias
	self._dropout = dropout
	self._scale = math.sqrt(float(key_dim))
	self._kernel_initializer = kernel_initializer
	self._bias_initializer = bias_initializer

  def build(self, input_shape: tf.TensorShape) -> None:
	input_dim = input_shape[-1]
	free_dims = input_shape.rank - 1
	common_kwargs = dict(
		kernel_initializer=self._kernel_initializer,
		bias_initializer=self._bias_initializer)

	einsum_equation, bias_axes, output_rank = _build_proj_equation(
		free_dims, bound_dims=1, output_dims=2)
	self.query_dense = tf.keras.layers.experimental.EinsumDense(
		einsum_equation,
		output_shape=_get_output_shape(output_rank - 1,
									   [self._num_heads, self._key_dim]),
		bias_axes=bias_axes if self._use_bias else None,
		**common_kwargs)

	if self._attn_type == "multi_head":
	  num_heads = self._num_heads
	elif self._attn_type == "multi_query":
	  num_heads = 1
	else:
	  raise ValueError(
		  "{} is not a recognized attention type".format(self._attn_type))
	self.key_dense = tf.keras.layers.experimental.EinsumDense(
		einsum_equation,
		output_shape=_get_output_shape(output_rank - 1,
									   [num_heads, self._key_dim]),
		bias_axes=bias_axes if self._use_bias else None,
		**common_kwargs)
	self.value_dense = tf.keras.layers.experimental.EinsumDense(
		einsum_equation,
		output_shape=_get_output_shape(output_rank - 1,
									   [num_heads, self._key_dim]),
		bias_axes=bias_axes if self._use_bias else None,
		**common_kwargs)

	self._dot_product_equations = []
	self._combine_equations = []
	self.softmax_layers = []
	for attn_axes in self._attn_axes:
	  attn_axes = tuple(attn_axes)
	  (dot_product_equation, combine_equation,
	   attn_scores_rank) = _build_attention_equation(output_rank, attn_axes)
	  norm_axes = tuple(
		  range(attn_scores_rank - len(attn_axes), attn_scores_rank))
	  self._dot_product_equations.append(dot_product_equation)
	  self._combine_equations.append(combine_equation)
	  self.softmax_layers.append(tf.keras.layers.Softmax(axis=norm_axes))

	output_shape = [input_dim]
	einsum_equation, bias_axes, output_rank = _build_proj_equation(
		free_dims, bound_dims=2, output_dims=len(output_shape))
	self.output_dense = tf.keras.layers.experimental.EinsumDense(
		einsum_equation,
		output_shape=_get_output_shape(output_rank - 1, output_shape),
		bias_axes=bias_axes if self._use_bias else None,
		**common_kwargs)

	self.dropout_layer = tf.keras.layers.Dropout(self._dropout)
	super().build(input_shape)

  def call(self,
		   queries: tf.Tensor,
		   values: tf.Tensor,
		   training: Optional[bool] = None) -> tf.Tensor:
	queries = self.query_dense(queries)
	keys = self.key_dense(values)
	values = self.value_dense(values)
	if self._attn_type == "multi_query":
	  keys = tf.repeat(keys, [self._num_heads], axis=-2)
	  values = tf.repeat(values, [self._num_heads], axis=-2)

	num_axes = len(self._attn_axes)
	queries = tf.split(queries, num_or_size_splits=num_axes, axis=-2)
	keys = tf.split(keys, num_or_size_splits=num_axes, axis=-2)
	values = tf.split(values, num_or_size_splits=num_axes, axis=-2)

	outputs = []
	for i in range(num_axes):
	  attn_scores = tf.einsum(self._dot_product_equations[i], keys[i],
							  queries[i]) / self._scale
	  attn_scores = self.softmax_layers[i](attn_scores)
	  attn_scores = self.dropout_layer(attn_scores, training=training)
	  outputs.append(
		  tf.einsum(self._combine_equations[i], attn_scores, values[i]))

	outputs = tf.concat(outputs, axis=-2)
	outputs = self.output_dense(outputs)
	return outputs



class Block(tf.keras.layers.Layer):
  """Aattention block."""

  def __init__(self,
			   attn_axes: List[List[int]],
			   num_heads: int = 4,
			   dropout: float = 0.0,
			   attn_dropout: float = 0.0,
			   attn_type: Text = "multi_head",
			   norm_type: Text = "layer",
			   activation: Callable[..., tf.Tensor] = tf.nn.gelu,
			   kernel_initializer="glorot_uniform",
			   bias_initializer="zeros",
			   **kwargs: Any) -> None:
	"""Initializer.
	Args:
	  attn_axes: A list for the list of axes over which the attention is
		applied.
	  num_heads: An integer for the number of attention heads.
	  dropout: A float for the dropout rate for MLP.
	  attn_dropout: A float for the dropout for attention.
	  attn_type: A string for attention type ("multi_head" or "multi_query").
	  norm_type: A string for the type of normalization.
	  activation: Activation function.
	  kernel_initializer: Initialization function of dense kenrels.
	  bias_initializer: Initialization function of dense biases.
	  **kwargs: Additional arguments for `tf.keras.layers.Layer`.
	"""
	super().__init__(**kwargs)
	self._attn_axes = attn_axes
	self._num_heads = num_heads
	self._dropout = dropout
	self._attn_dropout = attn_dropout
	self._attn_type = attn_type
	self._norm_type = norm_type
	self._activation = activation
	self._kernel_initializer = kernel_initializer
	self._bias_initializer = bias_initializer

  def build(self, input_shapes: Union[tf.TensorShape, Tuple[tf.TensorShape,
															tf.TensorShape]]):
	if isinstance(input_shapes, tuple):
	  input_dim = input_shapes[0][-1]
	else:
	  input_dim = input_shapes[-1]

	common_kwargs = dict(
		kernel_initializer=self._kernel_initializer,
		bias_initializer=self._bias_initializer)

	self.attention_layer = MultiAxisAttention(
		num_heads=self._num_heads,
		key_dim=input_dim // self._num_heads,
		attn_axes=self._attn_axes,
		attn_type=self._attn_type,
		dropout=self._attn_dropout,
		**common_kwargs)

	self.norm = make_norm_layer(self._norm_type)
	self.dropout_layer = tf.keras.layers.Dropout(self._dropout)
	self.mlp_block = MLP(
		dropout=self._dropout,
		norm_type=self._norm_type,
		activation=self._activation,
		**common_kwargs)
	super().build(input_shapes)

  def call(self,
		   inputs: Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]],
		   training: Optional[bool] = None) -> tf.Tensor:
	if isinstance(inputs, tuple):
	  queries, values = inputs
	else:
	  queries = inputs
	  values = None

	outputs = self.norm(queries, training=training)
	if values is None:
	  values = outputs

	outputs = self.attention_layer(outputs, values, training=training)
	outputs = self.dropout_layer(outputs, training=training)
	outputs = outputs + queries

	outputs = self.mlp_block(outputs, training=training)
	return outputs




class Generator(tf.keras.Model):
	def __init__(self) -> None:
		"""
		Args:
			output_size: An integer for the output size.
			output_dim: An integer for the output channel dimension.
			attn_type: A string for attention type ("multi_head" or "multi_query").
			norm_type: A string for the type of normalization.
			activation: Activation function.
		**kwargs: Additional arguments for `tf.keras.layers.Layer`.
		"""
		super().__init__()

	def call(self, inputs: tf.Tensor, training: Optional[bool] = None):
		"""Computes a forward pass of the generator block.
		Args:
			inputs: The input latent codes with the shape (batch_size, channel_dim).
			training: Boolean, whether training or not.
		Returns:
			The output feature map.
		"""
		outputs = self.dense_layer(inputs)
		embeddings = self.embedding_layer(inputs, training=training)
		images = None

		for i in range(self._num_blocks):
			outputs = self.position_embeddings[i](outputs)
			patch_size = self._patch_size_per_block[i]
			if patch_size is not None:
				grid_size = outputs.shape[2] // patch_size
				outputs = block_voxels(outputs, patch_size)
				outputs = self.blocks[i]((outputs, embeddings), training=training)
				outputs = unblock_voxels(outputs, grid_size, patch_size)
			else:
				outputs = self.blocks[i]((outputs, embeddings), training=training)

			if self.to_rgb_layers[i] is not None:
				images = self.to_rgb_layers[i](outputs, images, training=training)

			if i < self._num_blocks - 1:
				outputs = self.umsamplings[i](outputs)

		return images







class tGAN(tf.keras.Model):
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



