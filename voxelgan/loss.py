import tensorflow as tf

loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real, generated):
    real_loss = loss_obj(tf.ones_like(real), real)
    generated_loss = loss_obj(tf.zeros_like(generated), generated)
    total_loss = real_loss + generated_loss
    return total_loss

def generator_loss(generated):
    return loss_obj(tf.ones_like(generated), generated)


class Loss:
    def __init__(self):
        self.discriminator_loss = discriminator_loss()
        self.generator_loss = generator_loss()



