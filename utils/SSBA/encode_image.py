# Source: https://github.com/SCLBD/ISSBA

import bchlib
import glob
import os
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
# import tensorflow.contrib.image
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants
import tensorflow.compat.v1 as tfc
from tqdm import tqdm

BCH_POLYNOMIAL = 137
BCH_BITS = 5
MODEL_PATH = './trigger/imagenet_encoder' # imagenet
IMAGE_DIR_BD = './utils/SSBA/saved_images_bd/'

class bd_generator:
    def __init__(self):

        # load model
        self.sess = tfc.InteractiveSession(graph=tf.Graph())
        self.model = tfc.saved_model.loader.load(self.sess, [tag_constants.SERVING], MODEL_PATH)

        if IMAGE_DIR_BD is not None:
            if not os.path.exists(IMAGE_DIR_BD):
                os.makedirs(IMAGE_DIR_BD)

    def generate_bdImage(self, image):
        # print("bd_generator class: calling generate_bdImage method.")
        sess = self.sess
        model = self.model

        secret = 'a'
        input_secret_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs[
            'secret'].name
        input_image_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs[
            'image'].name
        input_secret = tfc.get_default_graph().get_tensor_by_name(input_secret_name)
        input_image = tfc.get_default_graph().get_tensor_by_name(input_image_name)

        output_stegastamp_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs[
            'stegastamp'].name
        output_residual_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs[
            'residual'].name
        output_stegastamp = tfc.get_default_graph().get_tensor_by_name(output_stegastamp_name)
        output_residual = tfc.get_default_graph().get_tensor_by_name(output_residual_name)

        width = 224
        height = 224

        bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)

        if len(secret) > 7:
            print('Error: Can only encode 56bits (7 characters) with ECC')
            return

        data = bytearray(secret + ' ' * (7 - len(secret)), 'utf-8')
        ecc = bch.encode(data)
        packet = data + ecc

        packet_binary = ''.join(format(x, '08b') for x in packet)
        secret = [int(x) for x in packet_binary]
        secret.extend([0, 0, 0, 0])

        size = (width, height)
        # print("image.shape =", image.shape)
        image = Image.fromarray(image, mode='RGB')
        image = np.array(ImageOps.fit(image, size), dtype=np.float32)
        image /= 255.
        # print("secret has shape", len(secret))
        # print("packet has length", len(packet))

        feed_dict = {input_secret: [secret],
                     input_image: [image]}
        # print(feed_dict)
        hidden_img, residual = sess.run([output_stegastamp, output_residual], feed_dict=feed_dict)
        rescaled = (hidden_img[0] * 255).astype(np.uint8)
        raw_img = (image * 255).astype(np.uint8)
        residual = residual[0] + .5
        residual = (residual * 255).astype(np.uint8)

        # save_name = filename.split('/')[-1].split('.')[0]
        # im = Image.fromarray(np.array(rescaled))
        # im = im.resize((112,112))
        # im.save(args.save_dir + '/'+save_name+'.jpg')
        # print("rescaled is of shape", rescaled.shape)
        return rescaled

