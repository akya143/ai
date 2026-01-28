import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os
np.random.seed(42)
tf.random.set_seed(42)
(x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32')
x_train = (x_train - 127.5) / 127.5  # Normalize to [-1, 1]
x_train = np.expand_dims(x_train, axis=-1)  # Add channel dimension (1, height, width, 1)
latent_dim = 100  # Dimension of the latent space (noise)
batch_size = 64
epochs = 10000
save_interval = 1000
def build_generator():
    model = tf.keras.Sequential()
    model.add(layers.Dense(128 * 7 * 7, input_dim=latent_dim))  # Output enough features for reshape
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.Reshape((7, 7, 128)))  # 7x7 image with 128 filters
    model.add(layers.Conv2DTranspose(128, kernel_size=3, strides=2, padding='same'))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding='same'))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.BatchNormalization(momentum=0.8))

    model.add(layers.Conv2DTranspose(1, kernel_size=3, strides=1, padding='same', activation='tanh'))

    return model
def build_discriminator():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, kernel_size=3, strides=2, padding='same', input_shape=(28, 28, 1)))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dropout(0.25))
    model.add(layers.Conv2D(128, kernel_size=3, strides=2, padding='same'))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dropout(0.25))

    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))  # Output a single value (real/fake)

    return model

def build_gan(generator, discriminator):
    discriminator.trainable = False
    gan_input = layers.Input(shape=(latent_dim,))
    x = generator(gan_input)
    gan_output = discriminator(x)

    gan = tf.keras.Model(gan_input, gan_output)
    return gan
discriminator = build_discriminator()
discriminator.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5), loss='binary_crossentropy', metrics=['accuracy'])

generator = build_generator()
gan = build_gan(generator, discriminator)
gan.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5), loss='binary_crossentropy')
def train_gan(epochs, batch_size, save_interval):
    half_batch = batch_size // 2  # Half the batch size for real and fake
# Adversarial ground truths
    valid = np.ones((batch_size, 1))  # Change to match full batch size
    fake = np.zeros((batch_size, 1))  # Change to match full batch size

    for epoch in range(epochs):
        idx = np.random.randint(0, x_train.shape[0], half_batch)
        real_images = x_train[idx]
        noise = np.random.normal(0, 1, (half_batch, latent_dim))
        generated_images = generator.predict(noise)
        d_loss_real = discriminator.train_on_batch(real_images, valid[:half_batch])  # Use only half batch for real
        d_loss_fake = discriminator.train_on_batch(generated_images, fake[:half_batch])  # Use only half batch for fake
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = gan.train_on_batch(noise, valid)  # Use full batch for generator
        if epoch % save_interval == 0:
            print(f"{epoch}/{epochs} [D loss: {d_loss[0]} | D accuracy: {100 * d_loss[1]}] [G loss: {g_loss}]")
            save_generated_images(epoch)
def save_generated_images(epoch, examples=10, dim=(1, 10), figsize=(10, 1)):
    noise = np.random.normal(0, 1, (examples, latent_dim))
    generated_images = generator.predict(noise)
    generated_images = (generated_images + 1) / 2.0  # Rescale images to [0, 1]
    plt.figure(figsize=figsize)
    for i in range(examples):
        plt.subplot(dim[0], dim[1], i + 1)
        plt.imshow(generated_images[i, :, :, 0], cmap='gray')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(f"gan_generated_image_{epoch}.png")
plt.close()

# Train the GAN
train_gan(epochs, batch_size, save_interval)

