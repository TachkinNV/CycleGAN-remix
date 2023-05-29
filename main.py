import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix
from PIL import Image
import os
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output
import numpy as np

import argparse

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description='Process multiple image files.')

# Add the positional arguments
parser.add_argument('image_files', nargs='+', help='input image files')

# Parse the command-line arguments
args = parser.parse_args()

# Access the parsed arguments
image_files = args.image_files

# Now you can use the image_files list in your script
for image_file in image_files:
    print(image_file)

AUTOTUNE = tf.data.experimental.AUTOTUNE

BUFFER_SIZE = 1000
BATCH_SIZE = 1
IMG_WIDTH = 512
IMG_HEIGHT = 512

# normalizing the images to [-1, 1]
def normalize(image):
  image = (image / 127.5) - 1
  image = tf.image.resize(image, [512, 512])
  return image

def preprocess_image_train(image_path, _):
  image = tf.io.read_file(image_path)
  image = tf.image.decode_jpeg(image)
  image = tf.cast(image, tf.float32)
  image = normalize(image)
  return image

train_content = tf.data.Dataset.from_tensor_slices(([image_files[0]], [image_files[0]]))
train_content = train_content.map(preprocess_image_train, num_parallel_calls=AUTOTUNE)
train_content = train_content.shuffle(BUFFER_SIZE)
train_content = train_content.batch(1)

train_style = tf.data.Dataset.from_tensor_slices(([image_files[1]], [image_files[1]]))
train_style = train_style.map(preprocess_image_train, num_parallel_calls=AUTOTUNE)
train_style = train_style.shuffle(BUFFER_SIZE)
train_style = train_style.batch(1)

OUTPUT_CHANNELS = 3
generator_g = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
generator_f = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
discriminator_x = pix2pix.discriminator(norm_type='instancenorm', target=False)
discriminator_y = pix2pix.discriminator(norm_type='instancenorm', target=False)

LAMBDA = 10

loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real, generated):
  real_loss = loss_obj(tf.ones_like(real), real)

  generated_loss = loss_obj(tf.zeros_like(generated), generated)

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss * 0.5

def generator_loss(generated):
  return loss_obj(tf.ones_like(generated), generated)

def calc_cycle_loss(real_image, cycled_image):
  loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))  
  return LAMBDA * loss1

def identity_loss(real_image, same_image):
  loss = tf.reduce_mean(tf.abs(real_image - same_image))
  return LAMBDA * 0.5 * loss

generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

EPOCHS = 70

def generate_images(model, test_input):
  prediction = model(test_input)  
  #output_path = 'image_orig.jpg'
  #tf.keras.preprocessing.image.save_img(output_path, test_input[0])
  output_path = 'image_predicted.jpg'
  tf.keras.preprocessing.image.save_img(output_path, prediction[0])


@tf.function
def train_step(real_x, real_y):
  # persistent is set to True because the tape is used more than
  # once to calculate the gradients.
  with tf.GradientTape(persistent=True) as tape:
    # Generator G translates X -> Y
    # Generator F translates Y -> X.
    
    fake_y = generator_g(real_x, training=True)
    cycled_x = generator_f(fake_y, training=True)

    fake_x = generator_f(real_y, training=True)
    cycled_y = generator_g(fake_x, training=True)

    # same_x and same_y are used for identity loss.
    same_x = generator_f(real_x, training=True)
    same_y = generator_g(real_y, training=True)

    disc_real_x = discriminator_x(real_x, training=True)
    disc_real_y = discriminator_y(real_y, training=True)

    disc_fake_x = discriminator_x(fake_x, training=True)
    disc_fake_y = discriminator_y(fake_y, training=True)

    # calculate the loss
    gen_g_loss = generator_loss(disc_fake_y)
    gen_f_loss = generator_loss(disc_fake_x)
    
    total_cycle_loss = calc_cycle_loss(real_x, cycled_x) + calc_cycle_loss(real_y, cycled_y)
    
    # Total generator loss = adversarial loss + cycle loss
    total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_loss(real_y, same_y)
    total_gen_f_loss = gen_f_loss + total_cycle_loss + identity_loss(real_x, same_x)

    disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
    disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)
  
  # Calculate the gradients for generator and discriminator
  generator_g_gradients = tape.gradient(total_gen_g_loss, 
                                        generator_g.trainable_variables)
  generator_f_gradients = tape.gradient(total_gen_f_loss, 
                                        generator_f.trainable_variables)
  
  discriminator_x_gradients = tape.gradient(disc_x_loss, 
                                            discriminator_x.trainable_variables)
  discriminator_y_gradients = tape.gradient(disc_y_loss, 
                                            discriminator_y.trainable_variables)
  
  # Apply the gradients to the optimizer
  generator_g_optimizer.apply_gradients(zip(generator_g_gradients, 
                                            generator_g.trainable_variables))

  generator_f_optimizer.apply_gradients(zip(generator_f_gradients, 
                                            generator_f.trainable_variables))
  
  discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                                discriminator_x.trainable_variables))
  
  discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                discriminator_y.trainable_variables))
for epoch in range(EPOCHS):
  start = time.time()

  n = 0
  for image_x, image_y in tf.data.Dataset.zip((train_content, train_style)):
    train_step(image_x, image_y)
    if n % 10 == 0:
      print ('.', end='')
    n += 1

  clear_output(wait=True)
  # Using a consistent image so that the progress of the model
  # is clearly visible.
  generate_images(generator_g, next(iter(train_content)))

  print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1, time.time()-start))

print('Generating finished')
