# -*- coding: utf-8 -*-
"""
Yizhe Zhang

TextCNN
"""
## 152.3.214.203/6006

import os

GPUID = 1
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPUID)

import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib import layers
# from tensorflow.contrib import metrics
# from tensorflow.contrib.learn import monitors
from tensorflow.contrib import framework
from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.python.platform import tf_logging as logging
# from tensorflow.contrib.learn.python.learn.metric_spec import MetricSpec
import cPickle
import numpy as np
import os
import scipy.io as sio
from math import floor
import pdb

from model import *
from utils import prepare_data_for_cnn, prepare_data_for_rnn, get_minibatches_idx, normalizing, restore_from_save, \
    prepare_for_bleu, cal_BLEU, sent2idx
from denoise import *

# import tempfile
# from tensorflow.examples.tutorials.mnist import input_data

logging.set_verbosity(logging.INFO)
# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS


class Options(object):
    def __init__(self):
        self.fix_emb = False
        self.reuse_w = False
        self.reuse_cnn = False
        self.reuse_discrimination = True  # reuse cnn for discrimination
        self.restore = True
        self.tanh = True  # activation fun for the top layer of cnn, otherwise relu
        self.model = 'cnn_deconv'  # 'cnn_rnn', 'rnn_rnn' , default: cnn_deconv

        self.permutation = 0
        self.substitution = 's'  # Deletion(d), Insertion(a), Substitution(s) and Permutation(p)

        self.W_emb = None
        self.cnn_W = None
        self.cnn_b = None
        self.maxlen = 253
        self.n_words = None
        self.filter_shape = 5
        self.filter_size = 300
        self.embed_size = 300
        self.lr = 1e-5
        self.layer = 3
        self.stride = [2, 2, 2]  # for two layer cnn/deconv , use self.stride[0]
        self.batch_size = 32
        self.max_epochs = 100
        self.n_gan = 900  # self.filter_size * 3
        self.L = 100

        self.save_path = "./save/" + "hotel_" + str(self.n_gan) + "_dim_" + self.model + "_" + self.substitution + str(
            self.permutation)
        self.log_path = "./log"
        self.print_freq = 100
        self.valid_freq = 100

        # batch norm & dropout
        self.batch_norm = False
        self.cnn_layer_dropout = False
        self.dropout = True
        self.dropout_ratio = 1.0
        self.is_train = True

        self.discrimination = False
        self.H_dis = 300

        self.sent_len = self.maxlen + 2 * (self.filter_shape - 1)
        self.sent_len2 = np.int32(floor((self.sent_len - self.filter_shape) / self.stride[0]) + 1)
        self.sent_len3 = np.int32(floor((self.sent_len2 - self.filter_shape) / self.stride[1]) + 1)
        self.sent_len4 = np.int32(floor((self.sent_len3 - self.filter_shape)/self.stride[2]) + 1)
        print ('Use model %s' % self.model)
        print ('Use %d conv/deconv layers' % self.layer)

    def __iter__(self):
        for attr, value in self.__dict__.iteritems():
            yield attr, value


def auto_encoder(x, x_org, opt, opt_t=None):
    # print x.get_shape()  # batch L
    if not opt_t: opt_t = opt
    x_emb, W_norm = embedding(x, opt)  # batch L emb
    x_emb = tf.expand_dims(x_emb, 3)  # batch L emb 1

    res = {}

    # cnn encoder
    if opt.layer == 4:
        H_enc = conv_model_4layer(x_emb, opt)
    elif opt.layer == 3:
        H_enc = conv_model_3layer(x_emb, opt)
    else:
        H_enc = conv_model(x_emb, opt)

    H_dec = H_enc
    # deconv decoder
    if opt.layer == 4:
        x_rec = deconv_model_4layer(H_dec, opt_t)  # batch L emb 1
    elif opt.layer == 3:
        x_rec = deconv_model_3layer(H_dec, opt_t)  # batch L emb 1
    else:
        x_rec = deconv_model(H_dec, opt_t)  # batch L emb 1
    print("Encoder len %d Decoder len %d Output len %d" % (
    x_emb.get_shape()[1], x_rec.get_shape()[1], x_org.get_shape()[1]))
    tf.assert_equal(x_rec.get_shape(), x_emb.get_shape())
    tf.assert_equal(x_emb.get_shape()[1], x_org.get_shape()[1])
    x_rec_norm = normalizing(x_rec, 2)  # batch L emb

    if opt.fix_emb:
        # cosine sim
        # Batch L emb
        loss = -tf.reduce_sum(x_rec_norm * x_emb)
        rec_sent = tf.argmax(tf.tensordot(tf.squeeze(x_rec_norm), W_norm, [[2], [1]]), 2)
        res['rec_sents'] = rec_sent


    else:
        x_temp = tf.reshape(x_org, [-1, ])
        prob_logits = tf.tensordot(tf.squeeze(x_rec_norm), W_norm, [[2], [1]])  # c_blv = sum_e x_ble W_ve

        prob = tf.nn.log_softmax(prob_logits * opt_t.L, dim=-1, name=None)
        rec_sent = tf.squeeze(tf.argmax(prob, 2))
        prob = tf.reshape(prob, [-1, opt_t.n_words])

        idx = tf.range(opt.batch_size * opt_t.sent_len)

        all_idx = tf.transpose(tf.stack(values=[idx, x_temp]))
        all_prob = tf.gather_nd(prob, all_idx)

        gen_temp = tf.cast(tf.reshape(rec_sent, [-1, ]), tf.int32)
        gen_idx = tf.transpose(tf.stack(values=[idx, gen_temp]))
        gen_prob = tf.gather_nd(prob, gen_idx)

        res['rec_sents'] = rec_sent

        res['gen_p'] = tf.exp(gen_prob[0:opt.sent_len])
        res['all_p'] = tf.exp(all_prob[0:opt.sent_len])

        if opt.discrimination:
            logits_real, _ = discriminator(x_org, W_norm, opt_t)
            prob_one_hot = tf.nn.log_softmax(prob_logits * opt_t.L * 100, dim=-1, name=None)
            logits_syn, _ = discriminator(tf.exp(prob_one_hot), W_norm, opt_t, is_prob=True, is_reuse=True)

            res['prob_r'] = tf.reduce_mean(tf.nn.sigmoid(logits_real))
            res['prob_f'] = tf.reduce_mean(tf.nn.sigmoid(logits_syn))

            loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits_real), logits=logits_real)) + \
                   tf.reduce_mean(
                       tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(logits_syn), logits=logits_syn))
        else:
            loss = -tf.reduce_mean(all_prob)
            
    tf.summary.scalar('loss', loss)

    train_op = layers.optimize_loss(
        loss,
        framework.get_global_step(),
        optimizer='Adam',
        learning_rate=opt.lr)
    return res, loss, train_op


def main():
    # global n_words
    # Prepare training and testing data
    loadpath = "./data/hotel_reviews.p"
    x = cPickle.load(open(loadpath, "rb"))
    train, val = x[0], x[1]
    wordtoix, ixtoword = x[2], x[3]
    train = [list(s) for s in train]
    val = [list(s) for s in val]
    opt = Options()
    opt.n_words = len(ixtoword) + 1
    ixtoword[opt.n_words - 1] = 'GO_'
    print dict(opt)
    print('Total words: %d' % opt.n_words)

    try:
        params = np.load('./param_g.npz')
        if params['Wemb'].shape == (opt.n_words, opt.embed_size):
            print('Use saved embedding.')
            opt.W_emb = params['Wemb']
        else:
            print('Emb Dimension mismatch: param_g.npz:' + str(params['Wemb'].shape) + ' opt: ' + str(
                (opt.n_words, opt.embed_size)))
            opt.fix_emb = False
    except IOError:
        print('No embedding file found.')
        opt.fix_emb = False

    with tf.device('/gpu:1'):
        x_ = tf.placeholder(tf.int32, shape=[opt.batch_size, opt.sent_len])
        x_org_ = tf.placeholder(tf.int32, shape=[opt.batch_size, opt.sent_len])
        res_, loss_, train_op = auto_encoder(x_, x_org_, opt)
        merged = tf.summary.merge_all()



    uidx = 0
    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    np.set_printoptions(precision=3)
    np.set_printoptions(threshold=np.inf)
    saver = tf.train.Saver()

    with tf.Session(config=config) as sess:
        train_writer = tf.summary.FileWriter(opt.log_path + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(opt.log_path + '/test', sess.graph)
        sess.run(tf.global_variables_initializer())
        if opt.restore:
            try:
                t_vars = tf.trainable_variables()
                loader = restore_from_save(t_vars, sess, opt)

            except Exception as e:
                print(e)
                print("No saving session, using random initialization")
                sess.run(tf.global_variables_initializer())

        for epoch in range(opt.max_epochs):
            print("Starting epoch %d" % epoch)
            kf = get_minibatches_idx(len(train), opt.batch_size, shuffle=True)
            for _, train_index in kf:
                uidx += 1
                sents = [train[t] for t in train_index]

                sents_permutated = add_noise(sents, opt)

                if opt.model != 'rnn_rnn' and opt.model != 'cnn_rnn':
                    x_batch_org = prepare_data_for_cnn(sents, opt)  # Batch L
                else:
                    x_batch_org = prepare_data_for_rnn(sents, opt)  # Batch L

                if opt.model != 'rnn_rnn':
                    x_batch = prepare_data_for_cnn(sents_permutated, opt)  # Batch L
                else:
                    x_batch = prepare_data_for_rnn(sents_permutated, opt, is_add_GO=False)  # Batch L

                _, loss = sess.run([train_op, loss_], feed_dict={x_: x_batch, x_org_: x_batch_org})

                if uidx % opt.valid_freq == 0:
                    opt.is_train = False
                    valid_index = np.random.choice(len(val), opt.batch_size)
                    val_sents = [val[t] for t in valid_index]

                    val_sents_permutated = add_noise(val_sents, opt)

                    if opt.model != 'rnn_rnn' and opt.model != 'cnn_rnn':
                        x_val_batch_org = prepare_data_for_cnn(val_sents, opt)
                    else:
                        x_val_batch_org = prepare_data_for_rnn(val_sents, opt)

                    if opt.model != 'rnn_rnn':
                        x_val_batch = prepare_data_for_cnn(val_sents_permutated, opt)
                    else:
                        x_val_batch = prepare_data_for_rnn(val_sents_permutated, opt, is_add_GO=False)

                    loss_val = sess.run(loss_, feed_dict={x_: x_val_batch, x_org_: x_val_batch_org})
                    print("Validation loss %f " % (loss_val))
                    res = sess.run(res_, feed_dict={x_: x_val_batch, x_org_: x_val_batch_org})
                    if opt.discrimination:
                        print ("Real Prob %f Fake Prob %f" % (res['prob_r'], res['prob_f']))
                    print "Val Orig :" + " ".join([ixtoword[x] for x in val_sents[0] if x != 0])
                    print "Val Perm :" + " ".join([ixtoword[x] for x in val_sents_permutated[0] if x != 0])
                    print "Val Recon:" + " ".join([ixtoword[x] for x in res['rec_sents'][0] if x != 0])

                    val_set = [prepare_for_bleu(s) for s in val_sents]
                    [bleu2s, bleu3s, bleu4s] = cal_BLEU([prepare_for_bleu(s) for s in res['rec_sents']], {0: val_set})
                    print 'Val BLEU (2,3,4): ' + ' '.join([str(round(it, 3)) for it in (bleu2s, bleu3s, bleu4s)])
                    summary = sess.run(merged, feed_dict={x_: x_val_batch, x_org_: x_val_batch_org})
                    test_writer.add_summary(summary, uidx)
                    opt.is_train = True


                if uidx % opt.print_freq == 0:
                    print("Iteration %d: loss %f " % (uidx, loss))
                    res = sess.run(res_, feed_dict={x_: x_batch, x_org_: x_batch_org})
                    print "Original     :" + " ".join([ixtoword[x] for x in sents[0] if x != 0])
                    print "Permutated   :" + " ".join([ixtoword[x] for x in sents_permutated[0] if x != 0])
                    if opt.model == 'rnn_rnn' or opt.model == 'cnn_rnn':
                        print "Reconstructed:" + " ".join([ixtoword[x] for x in res['rec_sents_feed_y'][0] if x != 0])
                    print "Reconstructed:" + " ".join([ixtoword[x] for x in res['rec_sents'][0] if x != 0])


                    summary = sess.run(merged, feed_dict={x_: x_batch, x_org_: x_batch_org})
                    train_writer.add_summary(summary, uidx)

            saver.save(sess, opt.save_path, global_step=epoch)


if __name__ == '__main__':
    main()
