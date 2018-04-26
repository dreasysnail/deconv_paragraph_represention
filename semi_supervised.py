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
    prepare_for_bleu, cal_BLEU, sent2idx, _clip_gradients_seperate_norm
from denoise import *

# import tempfile
# from tensorflow.examples.tutorials.mnist import input_data

logging.set_verbosity(logging.INFO)
# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS


# flags.DEFINE_string('train_dir', 'data', 'Directory to put the training data.')

class Options(object):
    def __init__(self):
        self.fix_emb = False
        self.reuse_w = True
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
        self.maxlen = 305
        self.n_words = None
        self.filter_shape = 5
        self.filter_size = 300
        self.multiplier = 1 # filtersize multiplier
        self.embed_size = 300
        self.lr = 2e-4
        self.layer = 3
        self.stride = [2, 2 ,2]  # for two layer cnn/deconv , use self.stride[0]
        self.batch_size = 64
        self.dis_batch_size = 64
        self.max_epochs = 1000
        self.n_gan = 500  # self.filter_size * 3
        self.L = 100

        self.optimizer = 'Adam'  # tf.train.AdamOptimizer(beta1=0.9) #'Adam' # 'Momentum' , 'RMSProp'
        self.clip_grad = None  # None  #100  #  20#
        self.attentive_emb = False
        self.decay_rate = 1

        self.save_path = "./save/yelp" #"./save/yelp_500_new"
        self.log_path = "./log"
        self.print_freq = 100
        self.valid_freq = 1000
        
        self.part_data = False
        #self.portion = float(sys.argv[1])  # 10%  1%
        self.portion = 1.0  # 10%  1%

        # batch norm & dropout
        self.batch_norm = False
        self.cnn_layer_dropout = False
        self.dropout_ratio = 0.5 # keep probability.
        self.rec_alpha = 1
        self.rec_decay_freq = 50
        self.pretrain_step = 50000

        self.discrimination = False
        self.H_dis = 300

        self.sent_len = self.maxlen + 2 * (self.filter_shape - 1)
        self.sent_len2 = np.int32(floor((self.sent_len - self.filter_shape) / self.stride[0]) + 1)
        self.sent_len3 = np.int32(floor((self.sent_len2 - self.filter_shape) / self.stride[1]) + 1)
        self.sent_len4 = np.int32(floor((self.sent_len3 - self.filter_shape) / self.stride[2]) + 1)
        print ('Use model %s' % self.model)
        print ('Use %d conv/deconv layers' % self.layer)

    def __iter__(self):
        for attr, value in self.__dict__.iteritems():
            yield attr, value


def semi_classifier(alpha, x, x_org, x_lab, y, dp_ratio, opt, opt_t=None):
    # print x.get_shape()  # batch L
    is_train = True
    if not opt_t: opt_t = opt
    x_lab_emb, W_norm = embedding(x_lab, opt)  # batch L emb
    x_emb = tf.nn.embedding_lookup(W_norm, x)
    x_emb = tf.expand_dims(x_emb, 3)  # batch L emb 1
    x_lab_emb = tf.expand_dims(x_lab_emb, 3)  # batch L emb 1
    x_lab_emb= tf.nn.dropout(x_lab_emb, dp_ratio)
    res = {}

    # cnn encoder
    H_enc, res = conv_encoder(x_emb, is_train, opt, res)
    H_lab_enc, res = conv_encoder(x_lab_emb, is_train, opt, res, is_reuse = True)
    H_dec = H_enc

    #H_lab_enc = tf.nn.dropout(H_lab_enc, opt.dropout_ratio)
    logits = classifier_2layer(H_lab_enc, opt, dropout = dp_ratio, prefix='classify', is_reuse=None)
    dis_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits))
    
    # calculate the accuracy
    prob = tf.nn.sigmoid(logits)

    # if opt.model == 'rnn_rnn':
    #     rec_loss, rec_sent_1, _ = seq2seq(x, x_org, opt)
    #     _, rec_sent_2, _ = seq2seq(x, x_org, opt, feed_previous=True, is_reuse=True)
    #     res['rec_sents_feed_y'] = rec_sent_1
    #     res['rec_sents'] = rec_sent_2

    # elif opt.model == 'cnn_rnn':
    #     # lstm decoder
    #     H_dec2 = tf.identity(H_dec)
    #     rec_loss, rec_sent_1, _ = lstm_decoder(H_dec, x_org, opt)  #

    #     _, rec_sent_2, _ = lstm_decoder(H_dec, x_org, opt, feed_previous=True, is_reuse=True)

    #     res['rec_sents_feed_y'] = rec_sent_1
    #     res['rec_sents'] = rec_sent_2

    # else:

    #     # deconv decoder
    rec_loss, res = deconv_decoder(H_dec, x_org, W_norm, is_train, opt_t, res)

    correct_prediction = tf.equal(tf.round(prob), tf.round(y))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # calculate the total loss
    loss = alpha * rec_loss + (1-alpha) * dis_loss

    tf.summary.scalar('loss', loss)
    tf.summary.scalar('rec_loss', rec_loss)
    tf.summary.scalar('dis_loss', dis_loss)
    summaries = [
        # "learning_rate",
        "loss"
        # "gradients",
        # "gradient_norm",
    ]
    global_step = tf.Variable(0, trainable=False)
    train_op = layers.optimize_loss(
        loss,
        global_step=global_step,
        # framework.get_global_step(),
        optimizer=opt.optimizer,
        clip_gradients=(lambda grad: _clip_gradients_seperate_norm(grad, opt.clip_grad)) if opt.clip_grad else None,
        #learning_rate_decay_fn=lambda lr, g: tf.train.exponential_decay(learning_rate=lr, global_step=g,
        #                                                                decay_rate=opt.decay_rate, decay_steps=3000),
        learning_rate=opt.lr,
        summaries=summaries
    )
    return res, dis_loss, rec_loss, loss, train_op, prob, accuracy


def run_model(opt, train_unlab_x, train_lab_x, train_lab, val_unlab_x, val_lab_x, val_lab, test, test_y, wordtoix, ixtoword):
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
        alpha_ = tf.placeholder(tf.float32, shape=())
        x_ = tf.placeholder(tf.int32, shape=[opt.batch_size, opt.sent_len])
        x_org_ = tf.placeholder(tf.int32, shape=[opt.batch_size, opt.sent_len])
        x_lab_ = tf.placeholder(tf.int32, shape=[opt.dis_batch_size, opt.sent_len])
        y_ = tf.placeholder(tf.float32, shape=[opt.dis_batch_size, 1])
        dp_ratio_ = tf.placeholder(tf.float32, name='dp_ratio_')
        res_, dis_loss_, rec_loss_, loss_, train_op, prob_, acc_ = semi_classifier(alpha_, x_, x_org_, x_lab_, y_, dp_ratio_, opt)
        merged = tf.summary.merge_all()

    uidx = 0
    max_val_accuracy = 0.0
    max_test_accuracy = 0.0
    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    # config = tf.ConfigProto(device_count={'GPU':0})
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

            kf = get_minibatches_idx(len(train_unlab_x), opt.batch_size, shuffle=True)
            for _, train_index in kf:
                uidx += 1
                
                if opt.rec_alpha > 0 and uidx > opt.pretrain_step and uidx % opt.rec_decay_freq == 0:
                    opt.rec_alpha -= 0.01
                    print "alpha: "+  str(opt.rec_alpha)
                
                sents = [train_unlab_x[t] for t in train_index]

                lab_index = np.random.choice(len(train_lab), opt.dis_batch_size, replace=False)
                lab_sents = [train_lab_x[t] for t in lab_index]
                batch_lab = [train_lab[t] for t in lab_index]
                batch_lab = np.array(batch_lab)
                batch_lab = batch_lab.reshape((len(batch_lab), 1))
                x_batch_lab = prepare_data_for_cnn(lab_sents, opt)

                sents_permutated = add_noise(sents, opt)

                if opt.model != 'rnn_rnn' and opt.model != 'cnn_rnn':
                    x_batch_org = prepare_data_for_cnn(sents, opt)  # Batch L
                else:
                    x_batch_org = prepare_data_for_rnn(sents, opt)  # Batch L

                if opt.model != 'rnn_rnn':
                    x_batch = prepare_data_for_cnn(sents_permutated, opt)  # Batch L
                else:
                    x_batch = prepare_data_for_rnn(sents_permutated, opt, is_add_GO=False)  # Batch L

                _, dis_loss, rec_loss, loss, acc = sess.run([train_op, dis_loss_, rec_loss_, loss_, acc_],
                                                       feed_dict= {alpha_: opt.rec_alpha, x_: x_batch, x_org_: x_batch_org, x_lab_: x_batch_lab, y_: batch_lab, dp_ratio_: opt.dropout_ratio})
                summary = sess.run(merged, feed_dict={alpha_: opt.rec_alpha, x_: x_batch, x_org_: x_batch_org, x_lab_: x_batch_lab, y_: batch_lab, dp_ratio_: opt.dropout_ratio})
                train_writer.add_summary(summary, uidx)


                if uidx % opt.print_freq == 0:
                    print("Iteration %d: dis_loss %f, rec_loss %f, loss %f, acc %f " % (uidx, dis_loss, rec_loss, loss, acc))

                if uidx % opt.valid_freq == 0:
                    #print("Iteration %d: dis_loss %f, rec_loss %f, loss %f " % (uidx, dis_loss, rec_loss, loss))
                    valid_index = np.random.choice(len(val_unlab_x), opt.batch_size)
                    val_sents = [val_unlab_x[t] for t in valid_index]
                    
                    val_sents_permutated = add_noise(val_sents, opt)

                    if opt.model != 'rnn_rnn' and opt.model != 'cnn_rnn':
                        x_val_batch_org = prepare_data_for_cnn(val_sents, opt)
                    else:
                        x_val_batch_org = prepare_data_for_rnn(val_sents, opt)

                    if opt.model != 'rnn_rnn':
                        x_val_batch = prepare_data_for_cnn(val_sents_permutated, opt)
                    else:
                        x_val_batch = prepare_data_for_rnn(val_sents_permutated, opt, is_add_GO=False)

                    rec_loss_val = sess.run(rec_loss_, feed_dict={x_: x_val_batch,
                                                                  x_org_: x_val_batch_org, dp_ratio_: 1.0})
                    print("Validation rec loss %f " % rec_loss_val)

                    kf_val = get_minibatches_idx(len(val_lab_x), opt.dis_batch_size, shuffle=False)
                    
                    prob_val = []
                    for _, val_ind in kf_val:
                        val_sents = [val_lab_x[t] for t in val_ind]
                        x_val_dis = prepare_data_for_cnn(val_sents, opt)
                        val_y = np.array([val_lab[t] for t in val_ind]).reshape((opt.dis_batch_size, 1))
                        val_prob = sess.run(prob_, feed_dict={x_lab_: x_val_dis, dp_ratio_: 1.0})
                        for x in val_prob:
                            prob_val.append(x)

                    ##### DON'T UNDERSTAND :error   val_index
                    # probs = []
                    # val_truth = []
                    # for i in range(len(val_lab)):
                    #     val_truth.append(val_lab[i])
                    #     if type(val_index[i]) != int:
                    #         temp = []
                    #         for j in val_index[i]:
                    #             temp.append(prob_val[j])
                    #         aver = sum(temp) * 1.0 / len(temp)
                    #         probs.append(aver)
                    #     else:
                    #         probs.append(prob_val[val_index[i]])

                    probs = []
                    val_truth = []
                    for i in range(len(prob_val)):
                        val_truth.append(val_lab[i])
                        probs.append(prob_val[i])        

                    count = 0.0
                    for i in range(len(probs)):
                        p = probs[i]
                        if p > 0.5:
                            if val_truth[i] == 1:
                                count += 1.0
                        else:
                            if val_truth[i] == 0:
                                count += 1.0

                    val_accuracy = count * 1.0 / len(probs)



                    print("Validation accuracy %f " % val_accuracy)

                    summary = sess.run(merged,
                                       feed_dict={alpha_: opt.rec_alpha, x_: x_val_batch, x_org_: x_val_batch_org, x_lab_: x_val_dis, y_: val_y, dp_ratio_: 1.0})
                    test_writer.add_summary(summary, uidx)

                    if val_accuracy >= max_val_accuracy:
                        max_val_accuracy = val_accuracy

                        kf_test = get_minibatches_idx(len(test), opt.dis_batch_size, shuffle=False)
                        prob_test = []
                        for _, test_ind in kf_test:
                            test_sents = [test[t] for t in test_ind]
                            x_test_batch = prepare_data_for_cnn(test_sents, opt)
                            test_prob = sess.run(prob_, feed_dict={x_lab_: x_test_batch, dp_ratio_: 1.0})
                            for x in test_prob:
                                prob_test.append(x)

                        probs = []
                        test_truth = []
                        for i in range(len(prob_test)):
                            test_truth.append(test_y[i])
                            probs.append(prob_test[i])

                        # probs = []
                        # test_truth = []
                        # for i in range(len(test_y)):
                        #     test_truth.append(test_y[i])
                        #     if type(test_index[i]) != int:
                        #         temp = [prob_test[j] for j in test_index[i]]
                        #         aver = sum(temp) * 1.0 / len(temp)
                        #         probs.append(aver)
                        #     else:
                        #         probs.append(prob_test[test_index[i]])

                        count = 0.0
                        for i in range(len(probs)):
                            p = probs[i]
                            if p > 0.5:
                                if test_truth[i] == 1.0:
                                    count += 1.0
                            else:
                                if test_truth[i] == 0.0:
                                    count += 1.0

                        test_accuracy = count * 1.0 / len(probs)

                        print("Test accuracy %f " % test_accuracy)

                        max_test_accuracy = test_accuracy

                def test_input(text):
                    x_input = sent2idx(text, wordtoix, opt)
                    res = sess.run(res_, feed_dict={x_: x_input, x_org_: x_batch_org})
                    print "Reconstructed:" + " ".join([ixtoword[x] for x in res['rec_sents'][0] if x != 0])


                    # res = sess.run(res_, feed_dict={x_: x_batch, x_org_: x_batch_org, is_train_: 1})
                    # print "Original     :" + " ".join([ixtoword[x] for x in sents[0] if x != 0])
                    # # print "Permutated   :" + " ".join([ixtoword[x] for x in sents_permutated[0] if x != 0])
                    # if opt.model == 'rnn_rnn' or opt.model == 'cnn_rnn':
                    #     print "Reconstructed:" + " ".join([ixtoword[x] for x in res['rec_sents_feed_y'][0] if x != 0])
                    # print "Reconstructed:" + " ".join([ixtoword[x] for x in res['rec_sents'][0] if x != 0])

                    # print "Probs:" + " ".join([ixtoword[res['rec_sents'][0][i]] +'(' +str(np.round(res['all_p'][i],2))+')' for i in range(len(res['rec_sents'][0])) if res['rec_sents'][0][i] != 0])


            print(opt.rec_alpha)
            print("Epoch %d: Max Valid accuracy %f" % (epoch, max_val_accuracy))
            print("Epoch %d: Max Test accuracy %f" % (epoch, max_test_accuracy))



            saver.save(sess, opt.save_path, global_step=epoch)


def main():
    # global n_words
    # Prepare training and testing data
    loadpath = "./data/yelp.p"
    x = cPickle.load(open(loadpath, "rb"))
    train, val, test = x[0], x[1], x[2]
    train_lab, val_lab, test_lab = x[3], x[4], x[5]
    wordtoix, ixtoword = x[6], x[7]
    
    train_unlab_x = [list(s) for s in train]
    train_lab_x = [list(s) for s in train]
    val_unlab_x = [list(s) for s in val]
    val_lab_x = [list(s) for s in val]
    test = [list(s) for s in test]

    train_lab = np.array(train_lab, dtype='float32')
    val_lab = np.array(val_lab, dtype='float32')
    test_lab = np.array(test_lab, dtype='float32')

    opt = Options()
    opt.n_words = len(ixtoword)
    print dict(opt)
    print('Total words: %d' % opt.n_words)

    run_model(opt, train_unlab_x, train_lab_x, train_lab, val_unlab_x, val_lab_x, val_lab,
              test, test_lab, wordtoix, ixtoword)




if __name__ == '__main__':
    main()
