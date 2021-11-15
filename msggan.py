'''
houses main classes and functions for MSG-GAN architecture.

currently being built to employ either MSG-StyleGAN or MSG-ProGAN at will,
though that may change.
'''

import tensorflow as tf
import tensorflow_addons as tfa
import dataset
import numpy as np
import random
import math
import csv
from pathlib import Path
from functools import partial
from tensorflow import keras as K
from tensorflow import experimental as tfexp
from tensorflow.keras import Model, backend
from tensorflow.keras.layers import Layer, Dense, Conv2D, Conv1D, Conv2DTranspose, Conv1DTranspose
from tensorflow.keras.layers import AveragePooling2D, AveragePooling1D, UpSampling2D, UpSampling1D
from tensorflow.keras.layers import Concatenate, LeakyReLU, Reshape, Input, Dot
from tensorflow.keras.metrics import Mean
from tensorflow.keras.optimizers import Adam, RMSprop, schedules
from tensorflow.keras.initializers import RandomNormal


# ---------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------
class DenseEQ(Dense):
    def __init__(self, channels, **kwargs):
        if 'kernel_initializer' in kwargs:
            raise Exception("Cannot override kernel_initializer")
        super().__init__(channels, kernel_initializer=RandomNormal(mean=0.0, stddev=2), **kwargs)

    # -----------------------------------------------------------------------------------------------
    def build(self, input_shape):
        super().build(input_shape)
        # The number of inputs
        n = np.product([int(val) for val in input_shape[1:]])
        # He initialisation constant
        self.c = np.sqrt(4/n)

    # -----------------------------------------------------------------------------------------------
    def call(self, inputs):
        output = backend.dot(inputs, self.kernel*self.c) # scale kernel
        if self.use_bias:
            output = tf.nn.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return output


# ---------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------
class Conv2DEQ(Conv2D):
    def __init__(self, channels, **kwargs):
        if 'kernel_initializer' in kwargs:
            raise Exception("Cannot override kernel_initializer")
        super().__init__(channels, kernel_initializer=RandomNormal(mean=0.0, stddev=2), **kwargs)

    # -----------------------------------------------------------------------------------------------
    def build(self, input_shape):
        super().build(input_shape)
        n = np.product([int(val) for val in input_shape[1:]])
        self.c = np.sqrt(4/n)

    # -----------------------------------------------------------------------------------------------
    def call(self, inputs):
        outputs = tf.nn.conv2d(inputs, self.kernel*self.c, strides=self.strides,
                padding=self.padding, data_format=self.data_format)

        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias, data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs


# ---------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------
class Conv2DTransposeEQ(Conv2DTranspose):
    def __init__(self, channels, **kwargs):
        self.channels = channels
        if 'kernel_initializer' in kwargs:
            raise Exception("Cannot override kernel_initializer")
        super().__init__(channels, kernel_initializer=RandomNormal(mean=0.0, stddev=2), **kwargs)

    # -----------------------------------------------------------------------------------------------
    def build(self, input_shape):
        super().build(input_shape)
        n = np.product([int(val) for val in input_shape[1:]])
        self.c = np.sqrt(4/n)

    # -----------------------------------------------------------------------------------------------
    def call(self, inputs):
        self.kernel *= self.c
        outputs = tf.nn.conv2d_transpose(inputs, self.kernel*self.c, 
                [32, inputs.shape[1], inputs.shape[1], self.channels], 
                strides=self.strides, padding=self.padding, data_format=self.data_format)

        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias, data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs


# ---------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------
class MiniBatchStdDev(Layer):
    def __init__(self, **kwargs):
        super(MiniBatchStdDev, self).__init__(**kwargs)

    # -----------------------------------------------------------------------------------------------
    def call(self, inputs):
        # inputs = tf.cast(inputs, tf.float32)
        mean = backend.mean(inputs, axis=0, keepdims=True)
        squ_diffs = backend.square(inputs - mean)
        mean_sq_diff = backend.mean(squ_diffs, axis=0, keepdims=True)
        mean_sq_diff += 1e-8
        stdev = backend.sqrt(mean_sq_diff)
        mean_pix = backend.mean(stdev, keepdims=True)
        shape = backend.shape(inputs)
        # this 'shape' thing on the next line might need to be adjusted for time series
        output = backend.tile(mean_pix, (shape[0], shape[1], shape[2], 1)) 
        combined = backend.concatenate([inputs, output], axis=-1)
        
        return combined


# ---------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------
class MSGGAN:
# ---------------------------------------------------------------------------------------------------
    def __init__(self, neurs=512, epochs=50000, batch_size=32, g_lr=1e-4, d_lr=1e-4, r1_gamma=10.0, 
                 epsilon=1e-3, eps2=1e-7, beta1=0.5, beta2=0.999, rho=0.9, momentum=0.0, endres=128,
                 outpath='', nchannels=3, ksize=3, usebias=False, startres=4, **kwargs):
        self.neurs      = neurs                         # perhaps we should think of a way to make this flexible
        self.epochs     = epochs
        self.batch_size = batch_size
        self.outpath    = outpath
        self.g_lr       = g_lr
        self.d_lr       = d_lr
        self.decay      = 0.0
        self.r1_gamma   = r1_gamma
        self.epsilon    = epsilon
        self.eps2       = eps2
        self.beta1      = beta1
        self.beta2      = beta2
        self.rho        = rho
        self.momentum   = momentum
        self.startres   = startres
        self.endres     = endres
        self.nchannels  = nchannels                      # 3 for rgb (must be overwritten for time series)
        self.nlabels    = 0
        self.kernels    = 0
        self.ksize      = ksize
        self.to_rgb     = []
        self.reals      = []
        self.usebias    = usebias
        self.g_opt      = self.declare_g_optimizer(opt='adam')   # (no argument) defaults to RMSProp. 
        self.d_opt      = self.declare_d_optimizer(opt='adam')   # we could add other optimizers later
        self.G          = self.generator()
        self.D          = self.discriminator()

        self.G.summary()
        self.D.summary()

    # -----------------------------------------------------------------------------------------------
    # the generator model
    def generator(self):
        log2            = int(math.log2(self.endres))
        neurlog2        = math.log2(self.neurs)
        self.kernels    = 0
        c_res           = 4             # current resolution of the fake images at each layer activation
        channels        = self.neurs
        self.to_rgb     = []

        # x = inputs = Input(shape=(channels,), name='g_input')
        inputs.append(Input(shape=(channels,), name='g_input'))
        x = inputs[0]
        x = Reshape((1, 1, channels), name='g_reshape')(x)
        x = Conv2DTranspose(channels, kernel_size=4, strides=4, activation=LeakyReLU(0.2),
                use_bias=False, padding='same', kernel_initializer='he_normal', name='g_conv1')(x)
        x = Conv2DTranspose(channels, kernel_size=3, strides=1, activation=LeakyReLU(0.2),
                    use_bias=False, padding='same', kernel_initializer='he_normal', name='g_conv2')(x)
        x = self.pixel_norm(x)

        for block in range(log2 - 2):
            y  = Conv2DTranspose(self.nchannels, kernel_size=1, strides=1)(x)
            self.to_rgb.append(y)
            x = UpSampling2D()(x)
            c_res *= 2
            if ((channels > 16) and (c_res > 32)):
                channels /= 2
            for i in range(2):
                x = Conv2DTranspose(channels, kernel_size=3, strides=1, activation=LeakyReLU(0.2),
                        use_bias=False, padding='same', kernel_initializer='he_normal')(x)
                x = self.pixel_norm(x)
        x = Conv2DTranspose(self.nchannels, kernel_size=1, strides=1, activation=LeakyReLU(0.2),
                use_bias=False, padding='same', kernel_initializer='he_normal')(x)
        self.to_rgb = np.flip(self.to_rgb, axis=0)    # puts to_rgb in backwards order for the discriminator
        outs = [x]
        for i in range(self.to_rgb.shape[0]):
            outs.append(self.to_rgb[i])
        self.kernels = channels

        return Model(inputs, outs, name='Generator')

    # -----------------------------------------------------------------------------------------------
    # the discrimnator model
    def discriminator(self):
        # getting a really stupid error right now and it might be related to the fact that we're 
        # sending the D an array/list rather than a tuple
        log2        = int(math.log2(self.endres))
        neurlog2    = int(math.log2(self.neurs))
        channels    = 2**(min(neurlog2, 14-log2))  # reduces starting channels s.t. we always end with 512
        iteration   = 1                     # tracks which batch of minis we should be feeding in
        inputs      = []
        c_res       = self.endres             # current resolution of the fake images at each layer activation
        channels    = self.kernels

        inputs.append(Input(shape=(self.endres, self.endres, self.nchannels)))
        x = inputs[0]
        for i in self.to_rgb:
            inputs.append(Input(shape=i.shape[1:], ragged=True))
        x = Conv2D(channels, kernel_size=1, strides=1, use_bias=False, padding='same')(x)

        # for anything bigger than endres == 32
        for block in range(log2 - 2):
            x = MiniBatchStdDev()(x)
            x = Conv2D(channels, kernel_size=3, strides=1, use_bias=False, activation=LeakyReLU(0.2),
                                                padding='same', kernel_initializer='he_normal')(x)
            if ((channels < self.neurs) and (c_res > 32)):
                channels *= 2
            x = Conv2D(channels, kernel_size=3, strides=1, use_bias=False, activation=LeakyReLU(0.2),
                                                padding='same', kernel_initializer='he_normal')(x)
            x = AveragePooling2D(strides=2)(x)
            c_res /= 2
            x = Concatenate(axis=-1)([x, inputs[iteration]])
            iteration += 1

        x = MiniBatchStdDev()(x)
        assert channels == self.neurs, 'channels do not == self.neurs at end of discriminator'
        x = Conv2D(channels, kernel_size=3, strides=1, use_bias=False, activation=LeakyReLU(0.2),
                                                    padding='same', kernel_initializer='he_normal')(x)
        x = Conv2D(channels, kernel_size=4, strides=4, use_bias=False, activation=LeakyReLU(0.2),
                                                    padding='same', kernel_initializer='he_normal')(x)
        x = Dense(1, activation='linear')(x)

        return Model(inputs, x, name='Discriminator')


    # -----------------------------------------------------------------------------------------------
    '''main training loop'''
    def train(self, reals):
        print('starting training loop')
        print('')
        self.batch_size = reals.batch_size
        self.endres     = reals.endres
        self.nchannels  = reals.nchannels
        self.nlabels    = reals.nlabels

        g_train_loss    = Mean()
        d_train_loss    = Mean()

        np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)   

        assert self.endres != 0, 'the size of endres was not correctly stored'
        assert self.nchannels != 0, 'the number of channels was not correctly stored'

        for epoch in range(self.epochs):
            reals.create_batch()
            d_loss  = self.train_d(reals)
            g_loss  = self.train_g()
            d_train_loss(d_loss)
            g_train_loss(g_loss)
            self.print_losses(epoch, g_train_loss.result(), d_train_loss.result())

            if epoch % 100 == 0:
                outputs = self.generate_samples()
                reals.report_objects(outputs[0], epoch)
                if ((epoch % 1000 == 0) and (epoch > 0)):
                    self.G.save_weights(self.outpath + '/weights/generator%05d.h5' %(epoch), True)
                    self.D.save_weights(self.outpath + '/weights/discriminator%05d.h5' %(epoch), True)

    # -----------------------------------------------------------------------------------------------
    def makewaves(self, reals):
        print('starting wave generation')
        print('')

        outputs = self.generate_samples()
        reals.save_objects(outputs[0], self.outpath)

    # -----------------------------------------------------------------------------------------------
    @tf.function
    def train_g(self):
        # print('entering train_g')
        noise  = np.random.normal(size=[self.batch_size, self.neurs], loc=0.0, scale=1.0)
        with tf.GradientTape() as t:
            fakes = self.G(noise, training=True)
            yfake = self.D(fakes, training=True)
            loss  = self.get_g_loss(yfake)
        grads = t.gradient(loss, self.G.trainable_variables)
        goods = tf.reduce_all(tf.stack([tf.reduce_all(tf.math.is_finite(g)) for g in grads]))
        tf.cond(goods, lambda: self.g_opt.apply_gradients(zip(grads, self.G.trainable_variables)), 
                                                                                false_fn=tf.no_op)

        return loss

    # -----------------------------------------------------------------------------------------------
    @tf.function
    def train_d(self, reals):
        # print('entering train_d')
        noise       = np.random.normal(size=[self.batch_size, self.neurs], loc=0.0, scale=1.0)
        with tf.GradientTape() as t:
            try:
                self.reals = [tf.constant(np.array(reals.objects[i])) 
                            for i in range(np.shape(reals.objects)[0])]
            except:
                pass
            fakes   = self.G(noise, training=False)
            yfake   = self.D(fakes, training=True)
            yreal   = self.D(self.reals, training=True)
            loss    = self.get_d_loss(yreal, yfake, self.reals, fakes)
        grads = t.gradient(loss, self.D.trainable_variables)
        goods = tf.reduce_all(tf.stack([tf.reduce_all(tf.math.is_finite(g)) for g in grads]))
        tf.cond(goods, lambda: self.d_opt.apply_gradients(zip(grads, self.D.trainable_variables)), 
                                                                                false_fn=tf.no_op)

        return loss

    # -----------------------------------------------------------------------------------------------
    def get_g_loss(self, yfake):
        assert yfake is not None

        return -yfake

    # -----------------------------------------------------------------------------------------------
    def get_d_loss(self, yreal, yfake, reals, fakes):
        assert yfake is not None
        assert yreal is not None
        reals   = [tf.cast(reals[i], tf.float32) for i in range(np.shape(reals)[0])]
        alpha   = tf.random.uniform([int(self.batch_size), 1, 1, 1], 0., 1.)
        loss    = yfake - yreal
        inter   = [reals[i] + (alpha * (fakes[i] - reals[i])) for i in range(np.shape(reals)[0])]
        discr   = partial(self.D, training=True)
        with tf.GradientTape() as t:
            t.watch(inter)
            pred = discr(inter)
        grads   = t.gradient(pred, [inter])[0]
        slopes  = [tf.sqrt(tf.reduce_sum(tf.square(grads[i]), axis=[1, 2, 3])) 
                                            for i in range(np.shape(reals)[0])]
        gp      = [tf.reduce_mean((slopes[i] - 1.)**2) for i in range(np.shape(reals)[0])]
        ep      = [tf.square(yreal[i]) for i in range(np.shape(reals)[0])]
        for i in range(np.shape(reals)[0]):
            loss   += gp[i]*self.r1_gamma + ep[i]*self.epsilon

        return loss

    # -----------------------------------------------------------------------------------------------
    def print_losses(self, epoch, g_train_loss, d_train_loss):
        if epoch == 0:
            print('LOSSES:')
            print('# ---------------------------------------------------------')
        print('epoch %d:   Generator: %f    Discriminator: %f    Difference: %f' 
                %(epoch, g_train_loss, d_train_loss, g_train_loss - d_train_loss))
        Path(self.outpath).mkdir(exist_ok=True)
        Path(self.outpath + '/weights').mkdir(exist_ok=True)
        with open(self.outpath + '/weights/losses.csv', 'a', newline='') as f:
	        writer = csv.writer(f)
	        writer.writerow([int(epoch), float(g_train_loss), float(d_train_loss)])

    # -----------------------------------------------------------------------------------------------
    @tf.function
    def generate_samples(self, m1=0, m2=0, l1=0, l2=0):
        noise = np.random.normal(size=[self.batch_size, self.neurs], loc=0.0, scale=1.0)
        # insert the labels into the noise
        # remember we will need to scale these meaningfully, and will need to insert gaussian noise
        # of an appropriate scale for each label
        if (m1!=0 and m2!=0 and l1!=0 and l2!=0):
            for i in range(len(noise)):
                noise[i].append(l1)
                noise[i].append(l2)
                noise[i].append(m1)
                noise[i].append(m2)
        # insert the wanted vals for mass

        return self.G(noise, training=False)

    # -----------------------------------------------------------------------------------------------
    # this can be overridden in subclasses if we find different architectures require different opts
    def declare_g_optimizer(self, opt=None):
        if opt == 'adam':
            return Adam(self.g_lr, self.beta1, self.beta2, self.eps2)
        elif opt == 'adam-wd': 
            # step = tf.Variable(0, trainable=False)
            # schedule = schedules.PiecewiseConstantDecay(
            #     [1000,2000,3000,4000,5000,6000,7000,8000,9000,10000],  
            #     [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0])
            # lr and wd can be a function or a tensor
            # lr = self.d_lr * schedule(step)
            lr = self.d_lr
            wd = lambda: self.decay
            return tfa.optimizers.AdamW(learning_rate=lr, weight_decay=wd, beta_1=self.beta1,
                                        beta_2=self.beta2, epsilon=self.eps2)
        else:
            return RMSProp(self.g_lr, self.rho, self.momentum, self.eps2)

    # -----------------------------------------------------------------------------------------------
    # this can be overridden in subclasses if we find different architectures require different opts
    def declare_d_optimizer(self, opt=None):
        if opt == 'adam':
            return Adam(self.d_lr, self.beta1, self.beta2, self.eps2)
        elif opt == 'adam-wd': 
            # step = tf.Variable(0, trainable=False)
            # schedule = schedules.PiecewiseConstantDecay( 
            #     [1000,2000,3000,4000,5000,6000,7000,8000,9000,10000],  
            #     [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0])
            # lr and wd can be a function or a tensor
            # lr = self.d_lr * schedule(step)
            lr = self.d_lr
            wd = lambda: self.decay
            return tfa.optimizers.AdamW(learning_rate=lr, weight_decay=wd, beta_1=self.beta1,
                                        beta_2=self.beta2, epsilon=self.eps2)
        else:
            return RMSProp(self.d_lr, self.rho, self.momentum, self.eps2)

    # -----------------------------------------------------------------------------------------------
    def pixel_norm(self, x, epsilon=1e-8):
        epsilon = tf.constant(epsilon, dtype=x.dtype)

        return x * tf.math.rsqrt(tf.reduce_mean(tf.square(x), axis=1, keepdims=True) + epsilon)


# ---------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------
class MiniBatchStdDevTs(Layer):
    def __init__(self, **kwargs):
        super(MiniBatchStdDevTs, self).__init__(**kwargs)

    # -----------------------------------------------------------------------------------------------
    def call(self, inputs):
        # inputs = tf.cast(inputs, tf.float32)
        mean = backend.mean(inputs, axis=0, keepdims=True)
        squ_diffs = backend.square(inputs - mean)
        mean_sq_diff = backend.mean(squ_diffs, axis=0, keepdims=True)
        mean_sq_diff += 1e-8
        stdev = backend.sqrt(mean_sq_diff)
        mean_pix = backend.mean(stdev, keepdims=True)
        shape = backend.shape(inputs)
        # this 'shape' thing on the next line might need to be adjusted for time series
        output = backend.tile(mean_pix, (shape[0], shape[1], 1)) 
        combined = backend.concatenate([inputs, output], axis=-1)
        
        return combined 


# ---------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------
# custom version of Conv1D that uses equalized learning rate
class Conv1DEQ(Conv1D):
    def __init__(self, channels, **kwargs):
        if 'kernel_initializer' in kwargs:
            raise Exception("Cannot override kernel_initializer")
        super().__init__(channels, kernel_initializer=RandomNormal(mean=0.0, stddev=2), **kwargs)

    # -----------------------------------------------------------------------------------------------
    def build(self, input_shape):
        super().build(input_shape)
        n = np.product([int(val) for val in input_shape[1:]])
        self.c = np.sqrt(4/n)

    # -----------------------------------------------------------------------------------------------
    def call(self, inputs):
        outputs = tf.nn.conv1d(inputs, self.kernel*self.c, stride=self.strides,
                 padding='SAME', data_format='NWC')

        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias, data_format='NWC')

        if self.activation is not None:
            return self.activation(outputs)
        return outputs


# ---------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------
class Conv1DTransposeEQ(Conv1DTranspose):
    def __init__(self, channels, out_shape, **kwargs):
        self.channels  = channels
        self.out_shape = out_shape
        if 'kernel_initializer' in kwargs:
            raise Exception("Cannot override kernel_initializer")
        super().__init__(channels, kernel_initializer=RandomNormal(mean=0.0, stddev=2), **kwargs)

    # -----------------------------------------------------------------------------------------------
    def build(self, input_shape):
        super().build(input_shape)
        n = np.product([int(val) for val in input_shape[1:]])
        self.c = np.sqrt(4/n)

    # -----------------------------------------------------------------------------------------------
    def call(self, inputs):
        outputs = tf.nn.conv1d_transpose(inputs, self.kernel*self.c, self.out_shape, 
                strides=self.strides, padding='SAME', data_format='NWC')

        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias, data_format='NWC')

        if self.activation is not None:
            return self.activation(outputs)
        return outputs


# ---------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------
# MSG_GAN for time series. overrides default implementation.
class MSG_GAN_ts(MSGGAN):
# ---------------------------------------------------------------------------------------------------
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.to_ts = []     # we'll need to convert to time series data rather than images

    # -----------------------------------------------------------------------------------------------
    # the generator model
    def generator(self):
        self.kernels    = 0
        iters           = int(math.log2(self.endres)) - int(math.log2(self.startres))
        c_res           = self.startres             # current resolution of the fake images at each layer activation
        channels        = self.neurs
        self.to_ts      = []
        inputs          = []

        inputs.append(Input(shape=(channels,)))
        x = inputs[0]
        x = Reshape((1, channels))(x)
        x = Conv1DTransposeEQ(channels, out_shape=[self.batch_size, self.startres, channels], 
                kernel_size=self.startres, strides=self.startres, activation=LeakyReLU(0.2),
                use_bias=self.usebias)(x)
        x = Conv1DTransposeEQ(channels, out_shape=[self.batch_size, x.shape[1], channels], 
                kernel_size=self.ksize, strides=1, activation=LeakyReLU(0.2),
                use_bias=self.usebias)(x)
        x = self.pixel_norm(x)

        for block in range(iters):
            y  = Conv1DTransposeEQ(self.nchannels, out_shape=[self.batch_size, x.shape[1], self.nchannels], 
                    kernel_size=1, strides=1, use_bias=self.usebias, activation=LeakyReLU(0.2))(x)
            self.to_ts.append(y)
            x = UpSampling1D()(x)
            c_res *= 2
            # if c_res > 2048:
            #     channels /= 2 
            for i in range(2):
                x = Conv1DTransposeEQ(channels, out_shape=[self.batch_size, x.shape[1], channels], 
                        kernel_size=self.ksize, strides=1, activation=LeakyReLU(0.2),
                        use_bias=self.usebias)(x)
                x = self.pixel_norm(x)
        x = Conv1DTransposeEQ(self.nchannels, 
                out_shape=[self.batch_size, x.shape[1], self.nchannels], 
                kernel_size=1, strides=1, activation=LeakyReLU(0.2),
                use_bias=self.usebias)(x)
        self.to_ts = np.flip(self.to_ts, axis=0)    # puts to_ts in backwards order for the discriminator
        outs = [x]
        for i in range(self.to_ts.shape[0]):
            outs.append(self.to_ts[i])
        self.kernels = channels
        # print(np.shape(outs[0]))

        return Model(inputs, outs, name='Generator')

    # -----------------------------------------------------------------------------------------------
    # the discrimnator model
    def discriminator(self):
        iteration   = 1                     # tracks which batch of minis we should be feeding in
        inputs      = []
        c_res       = self.endres           # current resolution of the fake images at each layer activation
        channels    = self.kernels
        iters       = int(math.log2(self.endres)) - int(math.log2(self.startres))

        inputs.append(Input(shape=(self.endres, self.nchannels)))
        x = inputs[0]
        for i in self.to_ts:
            inputs.append(Input(shape=i.shape[1:], ragged=True))
        x = Conv1DEQ(channels, kernel_size=1, strides=1, use_bias=self.usebias, activation=None)(x)

        for block in range(iters):
            x = MiniBatchStdDevTs()(x)
            x = Conv1DEQ(channels, kernel_size=self.ksize, strides=1, use_bias=self.usebias, 
                    activation=LeakyReLU(0.2))(x)
            # if c_res > 2048:
            #     channels *= 2
            x = Conv1DEQ(channels, kernel_size=self.ksize, strides=1, use_bias=self.usebias, 
                    activation=LeakyReLU(0.2))(x)
            x = AveragePooling1D(strides=2)(x)
            c_res /= 2
            x = Concatenate(axis=-1)([x, inputs[iteration]])
            iteration += 1

        x = MiniBatchStdDevTs()(x)
        assert channels == self.neurs, 'channels do not == self.neurs at end of discriminator'
        x = Conv1DEQ(channels, kernel_size=self.ksize, strides=1, use_bias=self.usebias, 
                activation=LeakyReLU(0.2))(x)
        x = Conv1DEQ(channels, kernel_size=self.startres, strides=self.startres, use_bias=self.usebias, 
                activation=LeakyReLU(0.2))(x)
        x = DenseEQ(1, activation='linear')(x)

        return Model(inputs, x, name='Discriminator')

    # -----------------------------------------------------------------------------------------------
    # ---:  if this was done correctly, this will apply a unique gp per-resolution rather
    #       than a single gp applied to all gradients
    def get_d_loss(self, yreal, yfake, reals, fakes):
        assert yfake is not None
        assert yreal is not None
        reals   = [tf.cast(reals[i], tf.float32) for i in range(np.shape(reals)[0])]
        alpha   = tf.random.uniform([int(self.batch_size), 1, 1], 0., 1.)
        loss    = yfake - yreal
        inter   = [reals[i] + (alpha * (fakes[i] - reals[i])) for i in range(np.shape(reals)[0])]
        discr   = partial(self.D, training=True)
        with tf.GradientTape() as t:
            t.watch(inter)
            pred = discr(inter)
        grads   = t.gradient(pred, [inter])[0]
        slopes  = [tf.sqrt(tf.reduce_sum(tf.square(grads[i]), axis=[1, 2])) 
                                            for i in range(np.shape(reals)[0])]
        gp      = [tf.reduce_mean((slopes[i] - 1.)**2) for i in range(np.shape(reals)[0])]
        ep      = [tf.square(yreal[i]) for i in range(np.shape(reals)[0])]
        for i in range(np.shape(reals)[0]):
            loss   += gp[i]*self.r1_gamma + ep[i]*self.epsilon

        return loss


# ---------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------
# MSG_GAN for time series. overrides default implementation.
class MSG_CcGAN_ts(MSG_GAN_ts):
# ---------------------------------------------------------------------------------------------------
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # -----------------------------------------------------------------------------------------------
    # the generator model
    def generator(self):
        self.kernels    = 0
        iters           = int(math.log2(self.endres)) - int(math.log2(self.startres))
        c_res           = self.startres             # current resolution of the fake images at each layer activation
        channels        = self.neurs
        self.to_ts      = []
        inputs          = []

        inputs.append(Input(shape=(channels+self.nlabels,)))
        x = inputs[0]
        x = Reshape((1, channels+self.nlabels))(x)
        x = Conv1DTransposeEQ(channels+self.nlabels, out_shape=[self.batch_size, self.startres, channels], 
                kernel_size=self.startres, strides=self.startres, activation=LeakyReLU(0.2),
                use_bias=self.usebias)(x)
        x = Conv1DTransposeEQ(channels, out_shape=[self.batch_size, x.shape[1], channels], 
                kernel_size=self.ksize, strides=1, activation=LeakyReLU(0.2),
                use_bias=self.usebias)(x)
        x = self.pixel_norm(x)

        for block in range(iters):
            y  = Conv1DTransposeEQ(self.nchannels, out_shape=[self.batch_size, x.shape[1], self.nchannels], 
                    kernel_size=1, strides=1, use_bias=self.usebias, activation=LeakyReLU(0.2))(x)
            self.to_ts.append(y)
            x = UpSampling1D()(x)
            c_res *= 2
            # if c_res > 2048:
            #     channels /= 2 
            for i in range(2):
                x = Conv1DTransposeEQ(channels, out_shape=[self.batch_size, x.shape[1], channels], 
                        kernel_size=self.ksize, strides=1, activation=LeakyReLU(0.2),
                        use_bias=self.usebias)(x)
                x = self.pixel_norm(x)
        x = Conv1DTransposeEQ(self.nchannels, 
                out_shape=[self.batch_size, x.shape[1], self.nchannels], 
                kernel_size=1, strides=1, activation=LeakyReLU(0.2),
                use_bias=self.usebias)(x)
        self.to_ts = np.flip(self.to_ts, axis=0)    # puts to_ts in backwards order for the discriminator
        outs = [x]
        for i in range(self.to_ts.shape[0]):
            outs.append(self.to_ts[i])
        self.kernels = channels
        # print(np.shape(outs[0]))

        return Model(inputs, outs, name='Generator')

    # -----------------------------------------------------------------------------------------------
    # the discrimnator model
    def discriminator(self):
        iteration   = 1                     # tracks which batch of minis we should be feeding in
        inputs      = []
        c_res       = self.endres           # current resolution of the fake images at each layer activation
        channels    = self.kernels
        iters       = int(math.log2(self.endres)) - int(math.log2(self.startres))

        inputs.append(Input(shape=(self.endres, self.nchannels)))
        x = inputs[0]
        for i in self.to_ts:
            inputs.append(Input(shape=i.shape[1:], ragged=True))
        x = Conv1DEQ(channels, kernel_size=1, strides=1, use_bias=self.usebias, activation=None)(x)

        for block in range(iters):
            x = MiniBatchStdDevTs()(x)
            x = Conv1DEQ(channels, kernel_size=self.ksize, strides=1, use_bias=self.usebias, 
                    activation=LeakyReLU(0.2))(x)
            # if c_res > 2048:
            #     channels *= 2
            x = Conv1DEQ(channels, kernel_size=self.ksize, strides=1, use_bias=self.usebias, 
                    activation=LeakyReLU(0.2))(x)
            x = AveragePooling1D(strides=2)(x)
            c_res /= 2
            x = Concatenate(axis=-1)([x, inputs[iteration]])
            iteration += 1

        x = MiniBatchStdDevTs()(x)
        assert channels == self.neurs, 'channels do not == self.neurs at end of discriminator'
        x = Conv1DEQ(channels, kernel_size=self.ksize, strides=1, use_bias=self.usebias, 
                activation=LeakyReLU(0.2))(x)
        x = Conv1DEQ(channels, kernel_size=self.startres, strides=self.startres, use_bias=self.usebias, 
                activation=LeakyReLU(0.2))(x)
        x = DenseEQ(1, activation='linear')(x)

        return Model(inputs, x, name='Discriminator')

    # -----------------------------------------------------------------------------------------------
    # ---:  if this was done correctly, this will apply a unique gp per-resolution rather
    #       than a single gp applied to all gradients
    def get_d_loss(self, yreal, yfake, reals, fakes):
        assert yfake is not None
        assert yreal is not None
        reals   = [tf.cast(reals[i], tf.float32) for i in range(np.shape(reals)[0])]
        alpha   = tf.random.uniform([int(self.batch_size), 1, 1], 0., 1.)
        loss    = yfake - yreal
        inter   = [reals[i] + (alpha * (fakes[i] - reals[i])) for i in range(np.shape(reals)[0])]
        discr   = partial(self.D, training=True)
        with tf.GradientTape() as t:
            t.watch(inter)
            pred = discr(inter)
        grads   = t.gradient(pred, [inter])[0]
        slopes  = [tf.sqrt(tf.reduce_sum(tf.square(grads[i]), axis=[1, 2])) 
                                            for i in range(np.shape(reals)[0])]
        gp      = [tf.reduce_mean((slopes[i] - 1.)**2) for i in range(np.shape(reals)[0])]
        ep      = [tf.square(yreal[i]) for i in range(np.shape(reals)[0])]
        for i in range(np.shape(reals)[0]):
            loss   += gp[i]*self.r1_gamma + ep[i]*self.epsilon

        return loss
