import numpy as np
from scipy.io import loadmat
from scipy.misc import imresize
from glob import glob
from tqdm import tqdm
import tensorflow as tf
import keras
import keras.backend as K
from keras.initializers import RandomNormal
from keras.layers import Dense, Activation, LeakyReLU, BatchNormalization
from keras.layers import Conv2D, Conv2DTranspose, Reshape, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


dirpath = '/media/qinritukou/7C4EB5374EB4EB52/LinuxData/data/'
img_size = 64

filenames = np.array(glob(dirpath + 'img_align_celeba/*.jpg'))
print(filenames.shape)


plt.figure(figsize=(10,8))
for i in range(20):
    img = plt.imread(filenames[i])
    plt.subplot(4, 5, i + 1)
    plt.imshow(img)
    plt.title(img.shape)
    plt.xticks([])
    plt.yticks([])
plt.tight_layout()
plt.show()

X_train, X_test = train_test_split(filenames, test_size=1000)

def load_image(filename, size=(img_size, img_size)):
    img = plt.imread(filename)
    # crop
    rows, cols = img.shape[:2]
    crop_r, crop_c = 150, 150
    start_row, start_col = (rows - crop_r) // 2, (cols - crop_c) // 2
    end_row, end_col = rows - start_row, cols - start_col
    img = img[start_row: end_row, start_col:end_col, :]
    # resize
    img = imresize(img, size)
    return img


plt.figure(figsize=(5, 4))
for i in range(20):
    img = load_image(filenames[i])
    plt.subplot(4, 5, i+1)
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
plt.tight_layout()
plt.show()


""""
As we will see later on, the generator is using tanh activation, for which we need to preprocess the image data into the range between -1 and 1.
"""
def preprocess(x):
    return (x/255)*2-1


def deprocess(x):
    return np.uint8((x+1)/2*255)

"""
Generator

The generator takes a latent sample (100 randomly generated numbers) and produces a color face image that should look like one from the CelebA dataset.
"""
def make_generator(input_size, leaky_alpha, init_stddev):
    # generates images in (32, 32, 3)
    model = Sequential()
    original_size = int(img_size / 8)
    model.add(Dense(original_size * original_size * 512, input_shape=(input_size,),
                    kernel_initializer=RandomNormal(stddev=init_stddev)))
    model.add(Reshape(target_shape=(original_size, original_size, 512)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=leaky_alpha))
    model.add(Conv2DTranspose(256, kernel_size=5, strides=2, padding='same',
                              kernel_initializer=RandomNormal(stddev=init_stddev))) # 8x8
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=leaky_alpha))
    model.add(Conv2DTranspose(128, kernel_size=5, strides=2, padding='same',
                              kernel_initializer=RandomNormal(stddev=init_stddev))) # 16x16
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=leaky_alpha))
    model.add(Conv2DTranspose(3, kernel_size=5, strides=2, padding='same',
                              kernel_initializer=RandomNormal(stddev=init_stddev))) # 32x32
    model.add(Activation('tanh'))
    return model


"""

Discriminator

The discriminator is a classifier to tell if the input image is real or fake.
"""
def make_discriminator(leaky_alpha, init_stddev):
    # classifies images in (32, 32, 3)
    model = Sequential()
    model.add(Conv2D(64, kernel_size=5, strides=2, padding='same',
                     kernel_initializer=RandomNormal(stddev=init_stddev),
                     input_shape=(img_size, img_size, 3)))  # 16x16
    model.add(LeakyReLU(alpha=leaky_alpha))
    model.add(Conv2D(128, kernel_size=5, strides=2, padding='same',
                     kernel_initializer=RandomNormal(stddev=init_stddev))) # 8x8
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=leaky_alpha))
    model.add(Conv2D(256, kernel_size=5, strides=2, padding='same',
                      kernel_initializer=RandomNormal(stddev=init_stddev))) # 4x4
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=leaky_alpha))
    model.add(Flatten())
    model.add(Dense(1, kernel_initializer=RandomNormal(stddev=init_stddev)))
    model.add(Activation('sigmoid'))
    return model


"""
DCGAN
"""
# beta_1 is the exponential decay rate for the 1st moment estimates in Adam optimizer
def make_DCGAN(sample_size,
               g_learning_rate,
               g_beta_1,
               d_learning_rate,
               d_beta_1,
               leaky_alpha,
               init_std):
    # generator
    generator = make_generator(sample_size, leaky_alpha, init_std)

    # discriminator
    discriminator = make_discriminator(leaky_alpha, init_std)
    discriminator.compile(optimizer=Adam(lr=d_learning_rate, beta_1=d_beta_1), loss='binary_crossentropy')

    # GAN
    gan = Sequential([generator, discriminator])
    gan.compile(optimizer=Adam(lr=g_learning_rate, beta_1=g_beta_1), loss='binary_crossentropy')

    return gan, generator, discriminator


"""
Utility functions
"""
def make_latent_samples(n_samples, sample_size):
    return np.random.normal(loc=0, scale=1, size=(n_samples, sample_size))


def make_trainable(model, trainable):
    for layer in model.layers:
        layer.trainable = trainable


def make_labels(size):
    return np.ones([size, 1]), np.zeros([size, 1])


def show_losses(losses):
    losses = np.array(losses)

    fig, ax = plt.subplots()
    plt.plot(losses.T[0], label='Discriminator')
    plt.plot(losses.T[1], label='Generator')
    plt.title("Validation Losses")
    plt.legend()
    plt.show()


def show_images(generated_images):
    n_images = len(generated_images)
    cols = 10
    rows = n_images // cols

    plt.figure(figsize=(10, 8))
    for i in range(n_images):
        img = deprocess(generated_images[i])
        ax = plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.show()


"""
Training DCGAN
"""
def train(
        g_learning_rate,
        g_beta_1,
        d_learning_rate,
        d_beta_1,
        leaky_alpha,
        init_std,
        smooth=0.1,
        sample_size=100,
        epochs=3,
        batch_size=128,
        eval_size=16,
        show_details=True
):
    """

    :param g_learning_rate: learning rate for the generator
    :param g_beta_1: the exponential decay rate for the 1st moment estimates in Adam optimizer
    :param d_learning_rate: learning rate for the discriminator
    :param d_beta_1: the exponential decay rate for the 1st moment estimates in Adam optimizer
    :param leaky_alpha:
    :param init_std:
    :param smooth:
    :param sample_size: latent sample size
    :param epochs:
    :param batch_size: train batch size
    :param eval_size: evaluate size
    :param show_details:
    :return:
    """

    # labels for the batch size and the test size
    y_train_real, y_train_fake = make_labels(batch_size)
    y_eval_real, y_eval_fake = make_labels(eval_size)

    # create a GAN, a generator and a discriminator
    gan, generator, discriminator = make_DCGAN(
        sample_size,
        g_learning_rate,
        g_beta_1,
        d_learning_rate,
        d_beta_1,
        leaky_alpha,
        init_std
    )

    losses = []
    for e in range(epochs):
        for i in tqdm(range(len(X_train) // batch_size)):
            # real CelebA images
            X_batch = X_train[i*batch_size:(i+1)*batch_size]
            X_batch_real = np.array([preprocess(load_image(filename)) for filename in X_batch])

            # latent samples and the generated digit images
            latent_samples = make_latent_samples(batch_size, sample_size)
            X_batch_fake = generator.predict_on_batch(latent_samples)

            # train the discriminator to detect real and fake images
            make_trainable(discriminator, True)
            discriminator.train_on_batch(X_batch_real, y_train_real * (1 - smooth))
            discriminator.train_on_batch(X_batch_fake, y_train_fake)

            # train the generator via GAN
            make_trainable(discriminator, False)
            gan.train_on_batch(latent_samples, y_train_real)


        # evaluate
        X_eval = X_test[np.random.choice(len(X_test), eval_size, replace=False)]
        X_eval_real = np.array([preprocess(load_image(filename)) for filename in X_eval])

        latent_samples = make_latent_samples(eval_size, sample_size)
        X_eval_fake = generator.predict_on_batch(latent_samples)


        d_loss = discriminator.test_on_batch(X_eval_real, y_eval_real)
        d_loss += discriminator.test_on_batch(X_eval_fake, y_eval_fake)
        g_loss = gan.test_on_batch(latent_samples, y_eval_real) # we want the fake to be realistic!


        losses.append((d_loss, g_loss))

        print("Epoch: {:>3}/{} Discriminator Loss: {:>6.4f} Generator Loss: {:>6.4f}".format(e+1, epochs, d_loss, g_loss))
        # show_images(X_eval_fake[:10])

        # save weights
        generator.save_weights('model_results/generator_weights', True)
        discriminator.save_weights('model_results/discriminator_weights', True)
        # save model
        generator.save('model_results/generator.hdf5', True)
        discriminator.save('model_results/discriminator.hdf5', True)

    # show the result
    if show_details:
        show_losses(losses)
        show_images(generator.predict(make_latent_samples(80, sample_size)))

    return generator


"""
training
"""
# train(g_learning_rate=0.0002,
#       g_beta_1=0.5,
#       d_learning_rate=0.0002,
#       d_beta_1=0.5,
#       leaky_alpha=0.2,
#       init_std=0.02)


train(g_learning_rate=0.0001,
      g_beta_1=0.5,
      d_learning_rate=0.001,
      d_beta_1=0.5,
      leaky_alpha=0.2,
      init_std=0.02)
