from __future__ import print_function, division


from keras_contrib.layers.normalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers import Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import losses
from keras.utils import to_categorical
from keras.utils import to_categorical
import keras.backend as K
import tensorflow as tf
import keras


from sklearn.model_selection import train_test_split
from glob import glob
from tqdm import tqdm
from scipy.misc import imresize
import numpy as np
import matplotlib.pyplot as plt


dirpath = '../../../../data/'
img_size = 32
filenames = np.array(glob(dirpath + 'fbp/Images/*.jpg'))
print(filenames.shape)

X_train, X_test = train_test_split(filenames, test_size=1000)


def load_image(filename, size=(img_size, img_size)):
    img = plt.imread(filename)
    # Crop
    rows, cols = img.shape[:2]
    crop_r, crop_c = 150, 150
    start_row, start_col = (rows - crop_r) // 2, (cols - crop_c) // 2
    end_row, end_col = rows - start_row, cols - start_col
    img = img[start_row: end_row, start_col: end_col, :]
    # resize
    img = imresize(img, size)
    return img


def preprocess(x):
    return (x / 255) * 2 - 1


def deprocess(x):
    return np.uint8((x + 1) / 2 * 255)


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


class CCGAN():
    def __init__(self):
        self.img_rows = img_size
        self.img_cols = img_size
        self.mask_height = 10
        self.mask_width = 10
        self.channels = 3
        self.num_classes = 0
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        # Number of filters in first layer of generator and dicriminator
        self.gf = 32
        self.df = 32

        optimizer = Adam(0.00001, 0.9)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=['mse', 'binary_crossentropy'],
                                   loss_weights=[1, 0],
                                   optimizer=optimizer, metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        masked_img = Input(shape=self.img_shape)
        gen_img = self.generator(masked_img)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The valid tasks generated images as input and determines validity
        valid, _ = self.discriminator(gen_img)

        # The combined model (stacked generator and discriminator( takes
        # masked_img as input => generates images => determines validity
        self.combined = Model(masked_img, valid)
        self.combined.compile(loss='mse', optimizer=optimizer)

    def build_generator(self):
        """U-Net Generator"""
        def conv2d(layer_input, filters, f_size=4, bn=True):
            """
            Layers used during downsampling
            :param layer_input:
            :param filters:
            :param f_size: kernel_size
            :param bn:
            :return:
            """
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            """Layers used during sampleing"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = BatchNormalization(momentum=0.8)(u)
            u = Concatenate()([u, skip_input])
            return u

        img = Input(shape=self.img_shape)

        # Downsampling
        d1 = conv2d(img, self.gf, bn=False)
        d2 = conv2d(d1, self.gf * 2)
        d3 = conv2d(d2, self.gf * 4)
        d4 = conv2d(d3, self.gf * 8)

        # Upsampling
        u1 = deconv2d(d4, d3, self.gf * 4)
        u2 = deconv2d(u1, d2, self.gf * 2)
        u3 = deconv2d(u2, d1, self.gf)

        u4 = UpSampling2D(size=2)(u3)
        output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u4)

        return Model(img, output_img)

    def build_discriminator(self):

        img = Input(shape=self.img_shape)

        model = Sequential()
        model.add(Conv2D(self.df * 2, kernel_size=4, strides=2, padding='same', input_shape=self.img_shape))
        model.add(LeakyReLU(alpha=0.8))
        model.add(Conv2D(self.df * 4, kernel_size=4, strides=2, padding='same'))
        model.add(LeakyReLU(alpha=.2))
        model.add(InstanceNormalization())
        model.add(Conv2D(self.df * 8, kernel_size=4, strides=2, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(InstanceNormalization())

        model.summary()

        img = Input(shape=self.img_shape)
        features = model(img)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(features)

        label = Flatten()(features)
        label = Dense(self.num_classes + 1, activation='softmax')(label)

        return Model(img, [validity, label])

    def mask_randomly(self, imgs):
        y1 = np.random.randint(0, self.img_cols - self.mask_height, imgs.shape[0])
        y2 = y1 + self.mask_height
        x1 = np.random.randint(0, self.img_rows - self.mask_width, imgs.shape[0])
        x2 = x1 + self.mask_width

        masked_imgs = np.empty_like(imgs)
        for i, img in enumerate(imgs):
            masked_img = img.copy()
            _y1, _y2, _x1, _x2 = y1[i], y2[i], x1[i], x2[i]
            masked_img[_y1:_y2, _x1:_x2, :] = 0
            masked_imgs[i] = masked_img
        return masked_imgs

    def train(self, epochs, batch_size=128, sample_interval=50):
        # labels for the batch size and the test size
        y_train_real, y_train_fake = make_labels(batch_size)
        # y_eval_real, y_eval_fake = make_labels(eval_size)

        losses = []

        for epoch in range(epochs):
            for i in tqdm(range(len(X_train) // batch_size)):
                # real CelebA images
                X_batch = X_train[i * batch_size:(i + 1) * batch_size]
                X_batch_real = np.array([preprocess(load_image(filename)) for filename in X_batch])

                masked_img = self.mask_randomly(X_batch_real)

                # Generate batch_size of new images
                gen_imgs = self.generator.predict(X_batch_real)

                valid = np.ones((batch_size, 4, 4, 1))
                fake = np.ones((batch_size, 4, 4, 1))

                d_loss_real = self.discriminator.train_on_batch(X_batch_real, [valid, y_train_real])
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, [fake, y_train_fake])
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                g_loss = self.combined.train_on_batch(masked_img, valid)

                # Plot the progress
                print("%d [D loss: %f, op_acc: %.2f%% [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[4], g_loss), flush=True)

                # If at save interval => save generated image samples
                if epoch % sample_interval == 0:
                    # Select a random half batch of images
                    self.sample_images(epoch, X_batch_real)
                    self.save_model()

    def sample_images(self, epoch, imgs):
        r, c = 3, 6

        masked_imgs = self.mask_randomly(imgs)
        gen_imgs = self.generator.predict(masked_imgs)

        imgs = (imgs + 1.0) * 0.5
        masked_imgs = (masked_imgs + 1.0) * 0.5
        gen_imgs = (gen_imgs + 1.0) * 0.5

        gen_imgs = np.where(gen_imgs < 0, 0, gen_imgs)


        fig, axs = plt.subplots(r, c)
        for i in range(c):
            axs[0, i].imshow(imgs[i, :, :, :])
            axs[0, i].axis('off')
            axs[1, i].imshow(masked_imgs[i, :, :, :])
            axs[1, i].axis('off')
            axs[2, i].imshow(gen_imgs[i, :, :, :])
            axs[2, i].axis('off')
        fig.savefig("../../../../output/ccgan/images/%d.png" % epoch)
        plt.close()

    def save_model(self):

        def save(model, model_name):
            arch_path = "../../../../output/ccgan/saved_model/%s.json" % model_name
            weights_path = "../../../../output/ccgan/saved_model/%s_weights" % model_name
            model_path = "../../../../output/ccgan/saved_model/%s.hdf5" % model_name
            options = {
                "file_arch": arch_path,
                "file_weight": weights_path,
                "file_model": model_path
            }
            json_string = model.to_json()
            open(options['file_arch'], 'w').write(json_string)
            model.save_weights(options['file_weight'])
            model.save(options["file_model"])

        save(self.generator, "ccgan_generator")
        save(self.discriminator, "ccgan_discriminator")


if __name__ == '__main__':
    ccgan = CCGAN()
    ccgan.train(epochs=20, batch_size=32, sample_interval=1)
