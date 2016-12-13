
# coding: utf-8

# In[1]:

import os
import numpy as np
import scipy.misc
import vgg
import tensorflow as tf
import numpy as np
from sys import stderr
from functools import reduce



# In[2]:

CONTENT_WEIGHT = 5e0
STYLE_WEIGHT = 1e2
TV_WEIGHT = 1e2
LEARNING_RATE = 1e1
STYLE_SCALE = 1.0
ITERATIONS = 800
VGG_PATH = 'imagenet-vgg-verydeep-19.mat'
CONTENT_LAYER = 'relu4_2'
STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')


# In[3]:

def img_input(path):
    return scipy.misc.imread(path).astype(np.float)
def im_output(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    scipy.misc.imsave(path, img)


# In[4]:

def prisma_core(network, initial, content, styles, iterations, content_weight, style_weight,learning_rate):

    shape = (1,) + content.shape
    style_shape = (1,) + styles.shape
    content_features = {}
    style_features = {}
    

    # compute content features in feedforward mode
    with tf.Session() as sess:
        image = tf.placeholder('float', shape=shape)
        net, mean_pixel = vgg.net(network, image)
        content_raw = np.array([vgg.process(content, mean_pixel)])
        content_features[CONTENT_LAYER] = net[CONTENT_LAYER].eval(
                feed_dict={image: content_raw})

    # compute style features in feedforward mode
    
    with tf.Session() as sess:
        image = tf.placeholder('float', shape=style_shape)
        net, _ = vgg.net(network, image)
        style_raw = np.array([vgg.process(styles, mean_pixel)])
        for layer in STYLE_LAYERS:
            features = net[layer].eval(feed_dict={image: style_raw})
            features = np.reshape(features, (-1, features.shape[3]))
            features_T = np.transpose(features)
            gram = np.matmul(features_T, features) / features.size
            style_features[layer] = gram

    # make stylized image using backpropogation
    with tf.Graph().as_default():
        initial = np.array([vgg.process(initial, mean_pixel)])
        initial = initial.astype('float32')
        image = tf.Variable(initial)
        net, mean_pixel = vgg.net(network, image)

        # content loss
        content_loss = content_weight * (
                       tf.nn.l2_loss(
                       net[CONTENT_LAYER] - content_features[CONTENT_LAYER]) /
                       content_features[CONTENT_LAYER].size)
        # style loss
        style_loss = 0
        style_losses = []
        for style_layer in STYLE_LAYERS:
            layer = net[style_layer]
            _, h, w, n = map(lambda i: i.value, layer.get_shape())
            size = h * w * n
            feature_vgg = tf.reshape(layer, (-1, n))
            gram = tf.matmul(tf.transpose(feature_vgg), feature_vgg) / size
            style_gram = style_features[style_layer]
            style_losses.append(tf.nn.l2_loss(gram - style_gram) / style_gram.size)
        style_loss += style_weight * reduce(tf.add, style_losses)
        

        # overall loss
        total_loss = content_loss + style_loss #+ tv_loss

        # optimizer setup
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)
        
        saver = tf.train.Saver()
        

        # optimization
        final_loss = float('inf')
        final_img = None
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            for i in range(iterations):
                print 'Iteration ' + str(i + 1) + '/' + str(iterations)
                print '  content loss: ' +   str(content_loss.eval())
                print '    style loss: ' + str(style_loss.eval())
                #stderr.write('       tv loss: %g\n' % tv_loss.eval())
                print '    total loss: ' + str(total_loss.eval())
                train_step.run()
                summary_writer = tf.train.SummaryWriter('./modelfile', sess.graph)
                if i == iterations - 1:
                    save_path = saver.save(sess, "./modelfile/model.ckpt")
                    this_loss = total_loss.eval()
                    if this_loss < final_loss:
                        final_loss = this_loss
                        final_img = image.eval()
                    yield (
                        i,
                        vgg.output(final_img.reshape(shape[1:]), mean_pixel)
                    )


# In[5]:

content_path = "flowers.jpg"
style_path = "1882.jpg"
output_path = "mini_prisma_test.jpg"
content_image = img_input(content_path)
style_image = img_input(style_path)
initial = content_image


# In[6]:

for itr, image in prisma_core(
        network = VGG_PATH,
        initial = initial,
        content = content_image,
        styles = style_image,
        iterations = ITERATIONS,
        content_weight = CONTENT_WEIGHT,
        style_weight = STYLE_WEIGHT,
        learning_rate = LEARNING_RATE
    ):
        im_output(output_path, image)

