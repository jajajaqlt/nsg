# Copyright 2017 Rice University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function
import numpy as np
np.set_printoptions(threshold=np.inf)
import os
import sys

import argparse
import sys
import json
from operator import itemgetter
from multiprocessing import Pool
from datetime import datetime

import tensorflow as tf
from data_extraction.data_reader.data_loader import Loader
from trainer_vae.model import Model
from trainer_vae.utils import read_config, dump_config, add_igmm_ins
from trainer_vae.igmm_np import multi_d_igmm, lam, rha, beta, omega, alpha
from utilities.basics import dump_json, truncate_two_decimals, read_json
from utilities.logging import create_logger

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_" \
                                  "BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def train(clargs):
    config_file = os.path.join(clargs.continue_from, 'config.json') \
        if clargs.continue_from is not None \
        else clargs.config
    with open(config_file) as f:
        config = read_config(json.load(f))

    loader = Loader(clargs.data, config)
    print("loading data Current Time =", datetime.now().strftime("%H:%M:%S"))
    model = Model(config)
    print("create model Current Time =", datetime.now().strftime("%H:%M:%S"))

    # adds api, kw and type dicts with indices as keys for igmm help function
    try:
        igmm_api_dict = {y: x for x, y in config.vocab.api_dict.items()}
        igmm_type_dict = {y: x for x, y in config.vocab.type_dict.items()}
        igmm_kw_dict = {y: x for x, y in config.vocab.kw_dict.items()}
    except:
        import pdb; pdb.set_trace()
     
    logger = create_logger(os.path.join(clargs.save, 'loss_values.log'))
    logger.info('Process id is {}'.format(os.getpid()))
    logger.info('GPU device is {}'.format(os.environ["CUDA_VISIBLE_DEVICES"]))
    logger.info('Amount of data used is {}'.format(config.num_batches * config.batch_size))
    num_params = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
    logger.info('Number of params {}\n\t'.format(num_params))

    # import pdb; pdb.set_trace()

    with tf.compat.v1.Session(
            config=tf.compat.v1.ConfigProto(log_device_placement=False,
                                            allow_soft_placement=True)) as sess:

        saver = tf.compat.v1.train.Saver(tf.global_variables(), max_to_keep=3)
        tf.global_variables_initializer().run()

        print("initializing variables Current Time =", datetime.now().strftime("%H:%M:%S"))
        # restore model
        if clargs.continue_from is not None:
            vars_ = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            old_saver = tf.compat.v1.train.Saver(vars_)
            ckpt = tf.train.get_checkpoint_state(clargs.continue_from)
            old_saver.restore(sess, ckpt.model_checkpoint_path)
        
        print("restoring model Current Time =", datetime.now().strftime("%H:%M:%S"))

        prev_igmm = [[None, None, None] for _ in range(config.num_batches * config.batch_size)]

        # training
        pool = Pool(processes=40)
        for i in range(config.num_epochs):
            loader.reset_batches()
            avg_loss, avg_ast_loss, avg_kl_loss = 0., 0., 0.,
            avg_ast_gen_loss_concept, avg_ast_gen_loss_api, \
            avg_gen_loss_type, avg_gen_loss_clstype, avg_ast_gen_loss_var, \
            avg_ast_gen_loss_op, avg_ast_gen_loss_method = 0., 0., 0., 0., 0., 0., 0.

            print("Pool before Current Time = ", datetime.now().strftime("%H:%M:%S"))
            # pool = Pool(processes=40)
            print("Pool after Current Time = ", datetime.now().strftime("%H:%M:%S"))
            for b in range(config.num_batches):
                nodes, edges, targets, var_decl_ids, ret_reached,\
                node_type_number, \
                type_helper_val, expr_type_val, ret_type_val, \
                all_var_mappers, iattrib, \
                ret_type, fp_in, fields, \
                apicalls, types, keywords, method, classname, javadoc_kws,\
                    surr_ret, surr_fp, surr_method = loader.next_batch()
                # import pdb; pdb.set_trace()
                # will be a tensor for igmm_means
                # problem is that the igmm_matrix will have variable first dimension
                # use [?, 256] tensor
                feed_dict = dict()
                feed_dict.update({model.nodes: nodes, model.edges: edges, model.targets: targets})
                feed_dict.update({model.var_decl_ids: var_decl_ids,
                                  model.ret_reached: ret_reached,
                                  model.iattrib: iattrib,
                                  model.all_var_mappers: all_var_mappers})
                feed_dict.update({model.node_type_number: node_type_number})
                feed_dict.update({model.type_helper_val: type_helper_val, model.expr_type_val: expr_type_val,
                                  model.ret_type_val: ret_type_val})
                feed_dict.update({model.return_type: ret_type})
                feed_dict.update({
                    model.formal_param_inputs: fp_in
                })
                feed_dict.update({model.field_inputs: fields})
                feed_dict.update({model.apicalls: apicalls, model.types: types, model.keywords: keywords,
                                  model.method: method, model.classname: classname,
                                  model.javadoc_kws: javadoc_kws})
                feed_dict.update({
                    model.surr_ret: surr_ret,
                    model.surr_fp: surr_fp,
                    model.surr_method: surr_method
                })
                feed_dict.update({
                    model.encoder.ev_drop_rate: config.ev_drop_rate,
                    model.encoder.ev_miss_rate: config.ev_miss_rate,
                    model.decoder.program_decoder.ast_tree.drop_prob: config.decoder_drop_rate
                })
                # feed_dict.update({model.latent_vectors: latent_vectors})
                run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)

                # extract evidences // normal way
                print('extracting vectors')
                # everything is 128*10*256, except the ret 128*256, datatypes all float
                target_fp, target_field, target_ret, target_method, target_class, \
                target_jd, target_surr = sess.run(
                    [model.encoder.program_encoder.fp_mean_enc.reshaper,
                        model.encoder.program_encoder.field_enc.reshaper,
                        model.encoder.program_encoder.ret_mean_enc.latent_encoding,
                        model.encoder.program_encoder.method_enc.reshaper,
                        model.encoder.program_encoder.class_enc.reshaper,
                        model.encoder.program_encoder.jd_enc.reshaper,
                        model.encoder.program_encoder.surr_enc.layer3],
                    feed_dict=feed_dict)

                # initialize placeholder values
                precisions = []
                counts = []
                fp_inds, field_inds, ret_inds, method_inds, class_inds, jd_inds, surr_inds = [], [], [], [], [], [], []
                for j in range(config.max_means):
                    precisions.append(np.zeros((config.batch_size, config.latent_size), dtype=np.float32))
                    counts.append(np.zeros((config.batch_size), dtype=np.int32))
                    fp_inds.append(np.zeros((config.batch_size, config.input_fp_depth), dtype=np.int32))
                    field_inds.append(np.zeros((config.batch_size, config.max_fields), dtype=np.int32))
                    ret_inds.append(np.zeros((config.batch_size), dtype=np.int32))
                    method_inds.append(np.zeros((config.batch_size, config.max_camel_case), dtype=np.int32))
                    class_inds.append(np.zeros((config.batch_size, config.max_camel_case), dtype=np.int32))
                    jd_inds.append(np.zeros((config.batch_size, config.max_keywords), dtype=np.int32))
                    surr_inds.append(np.zeros((config.batch_size, config.max_keywords), dtype=np.int32))
                prefix_dict = {'fp':fp_inds, 'field':field_inds, 'ret':ret_inds, 'method':method_inds, 'class':class_inds, 'jd':jd_inds, 'surr':surr_inds}

                batch_igmm_input = []
                batch_igmm_params = []
                print('epoch {} batch {}'.format(i, b))
                print("Current Time =", datetime.now().strftime("%H:%M:%S"))
                for j in range(config.batch_size):
                    ins_labels = []
                    ins_nums = []
                    # 3. fp
                    add_igmm_ins(fp_in[j], target_fp[j], igmm_type_dict, 'fp_', ins_labels, ins_nums)
                    # 4. field
                    add_igmm_ins(fields[j], target_field[j], igmm_type_dict, 'field_', ins_labels, ins_nums)
                    # 5. ret
                    add_igmm_ins([ret_type[j]], [target_ret[j]], igmm_type_dict, 'ret_', ins_labels, ins_nums)
                    # 6. method
                    add_igmm_ins(method[j], target_method[j], igmm_kw_dict, 'method_', ins_labels, ins_nums)
                    # 7. class
                    add_igmm_ins(classname[j], target_class[j], igmm_kw_dict, 'class_', ins_labels, ins_nums)
                    # 8. jd
                    add_igmm_ins(javadoc_kws[j], target_jd[j], igmm_kw_dict, 'jd_', ins_labels, ins_nums)
                    # 9. surr
                    # add_igmm_ins(surr_method_first, target_surr[j], igmm_kw_dict, 'surr_', ins_labels, ins_nums)
                    add_igmm_ins([surr_ret[j], surr_method[j], surr_fp[j]], target_surr[j], [igmm_type_dict, igmm_kw_dict], 'surr_', ins_labels, ins_nums)
                    batch_igmm_input.append([ins_labels, ins_nums])
                    curr_idx = b * config.batch_size + j
                    str_idx = str(b) + '_' + str(j) + '_' + str(curr_idx)
                    igmm_samples = 30 if i == 0 else 3
                    prev_indicators, prev_means, prev_precs = prev_igmm[curr_idx]
                    batch_igmm_params.append([str_idx, config.latent_size, ins_nums, igmm_samples, prev_indicators, prev_means, prev_precs])
                
                # performance improvement, save batch_igmm_params
                # import pickle
                # with open('first_batch_testdata.pickle', 'wb') as f:
                #     pickle.dump(batch_igmm_params, f)
                # import sys
                # sys.exit(0)
                    
                remain_batch = config.batch_size
                while remain_batch > 0:
                    print("Remain_batch: ", remain_batch, "Current Time = ", datetime.now().strftime("%H:%M:%S"))
                    num_chunks = min(40, config.batch_size)
                    print("Current Time = ", datetime.now().strftime("%H:%M:%S"))
                    num_chunks = min(remain_batch, num_chunks)
                    print("Current Time = ", datetime.now().strftime("%H:%M:%S"))
                    # pool = Pool(processes=num_chunks)
                    # print("Current Time = ", datetime.now().strftime("%H:%M:%S"))
                    # import pdb; pdb.set_trace()
                    # result_list = []
                    # for bip in batch_igmm_params[:num_chunks]:
                    #     result = pool.apply_async(multi_d_igmm, bip)
                    #     result_list.append(result)
                    for result in pool.map(multi_d_igmm, batch_igmm_params[:num_chunks]):
                    # for result in result_list:
                        # import pdb; pdb.set_trace()
                        str_idx, indicators, means, precs = result
                        # strings
                        _, j, curr_idx = str_idx.split('_')
                        prev_igmm[int(curr_idx)] = [indicators, means, precs]
                        j = int(j)
                        ins_labels = batch_igmm_input[j][0]
                        pair_results = list(zip(indicators, ins_labels))
                    # curr_idx = b * config.batch_size + j
                    # if b == 0:
                    #     indicators, means, precs = multi_d_igmm(config.latent_size, ins_nums, 50)
                    # else:
                    #     prev_indicators, prev_means, prev_precs = prev_igmm[curr_idx]
                    #     indicators, means, precs = multi_d_igmm(config.latent_size, ins_nums, 3, prev_indicators, prev_means, prev_precs)
                    # prev_igmm[curr_idx] = [indicators, means, precs]

                    # pair_results = list(zip(indicators, ins_labels))

                        # fill the precisions
                        for k in range(min(len(precs), config.max_means)):
                            precisions[k][j,:] = precs[k]
                        
                        # fill the counts
                        for k in range(min(len(precs), config.max_means)):
                            count = indicators.count(k)
                            counts[k][j] = count

                        # fill the indicator matrices and vectors for rets
                        for (indicator, label) in pair_results:
                            if indicator >= config.max_means:
                                continue
                            labels = label.split('_')
                            index = int(labels[0])
                            prefix = labels[1]
                            if prefix != 'ret':
                                try:
                                    prefix_dict[prefix][indicator][j][index] = 1
                                except:
                                    import pdb; pdb.set_trace()
                            else:
                                # for ret, index is always 0
                                prefix_dict[prefix][indicator][j] = 1

                        sorted_cluster = sorted(pair_results, key=itemgetter(0), reverse=False)
                        # print((str(i)+'_'+str_idx))
                        # print(*sorted_cluster, sep='\n')
                        # logger.info(str(i)+'_'+str_idx)
                        # logger.info(sorted_cluster)
                    # pool.terminate()
                    # import pdb; pdb.set_trace()
                    remain_batch -= num_chunks
                    batch_igmm_params = batch_igmm_params[num_chunks:]

                # feed placeholders
                # feed_dict.update({model.latent_vectors: latent_vectors})
                feed_dict.update({model.encoder.program_encoder.lamda: lam,
                                  model.encoder.program_encoder.rha: rha})
                # feed precions, counts and indicators
                keys_and_values_pair = [(model.encoder.program_encoder.precisions, precisions),
                                        (model.encoder.program_encoder.counts, counts),
                                        (model.encoder.program_encoder.fp_mean_enc_indmats, fp_inds),
                                        (model.encoder.program_encoder.field_enc_indmats, field_inds),
                                        (model.encoder.program_encoder.ret_mean_enc_indvecs, ret_inds),
                                        (model.encoder.program_encoder.method_enc_indmats, method_inds),
                                        (model.encoder.program_encoder.class_enc_indmats, class_inds),
                                        (model.encoder.program_encoder.jd_enc_indmats, jd_inds),
                                        (model.encoder.program_encoder.surr_enc_indmats, surr_inds)]
                for (keys, values) in keys_and_values_pair:
                    dictionary = dict(zip(keys, values))
                    feed_dict.update(dictionary)

                for variable in dir():
                    print(variable, ': ', sys.getsizeof(eval(variable)))
                # import pdb; pdb.set_trace()
                # igmm // normal way
                # outputs indicators, means, precs
                # indicators: [#dps] [12], means&precs: [[#dim]+#gauss] [[256]^num_means]
                # find a real example: /home/letao/gibbs_sampler/012523temp_output_dir1k
                
                #1 ev_vec_extract original branch code
                #2 igmm original branch code
                #3 reparam-trick set_encoder.py: class KeywordEncoder
                
                # fake data here, could be later

                # run the optimizer
                loss, ast_loss, \
                ast_gen_loss_concept, ast_gen_loss_api, \
                ast_gen_loss_type, ast_gen_loss_clstype, ast_gen_loss_var, \
                ast_gen_loss_op, ast_gen_loss_method, \
                kl_loss, _, sigma = \
                    sess.run([model.loss, model.ast_gen_loss,
                              model.ast_gen_loss_concept, model.ast_gen_loss_api,
                              model.ast_gen_loss_type, model.ast_gen_loss_clstype,
                              model.ast_gen_loss_var, model.ast_gen_loss_op, model.ast_gen_loss_method,
                              model.KL_loss, model.train_op, model.encoder.program_encoder.sigmas], feed_dict=feed_dict)

                # if i % 10 == 0:
                #     return_alphas = sess.run(model.decoder.program_decoder.ast_tree.return_alphas_list, feed_dict=feed_dict)
                #     return_alphas_ndarray = np.array(return_alphas)
                #     return_alphas_ndarray_trans = np.transpose(return_alphas_ndarray, (1, 0, 2))
                #     # print(*return_alphas,sep='\n')
                #     # logger.info(return_alphas)
                #     # import pdb; pdb.set_trace()
                #     index_start = b * config.batch_size
                #     index_end = (b + 1) * config.batch_size - 1
                #     print('{}th epoch, {}th batch, index: {}-{}'.format(i, b, index_start, index_end))
                #     print(return_alphas_ndarray_trans)
                #     logger.info('{}th epoch, {}th batch, index: {}-{}'.format(i, b, index_start, index_end))
                #     logger.info(return_alphas_ndarray_trans)

                avg_loss += np.mean(loss)
                avg_ast_loss += np.mean(ast_loss)

                avg_ast_gen_loss_concept += np.mean(ast_gen_loss_concept)
                avg_ast_gen_loss_method += np.mean(ast_gen_loss_method)
                avg_ast_gen_loss_api += np.mean(ast_gen_loss_api)
                avg_gen_loss_type += np.mean(ast_gen_loss_type)
                avg_gen_loss_clstype += np.mean(ast_gen_loss_clstype)
                avg_ast_gen_loss_var += np.mean(ast_gen_loss_var)
                avg_ast_gen_loss_op += np.mean(ast_gen_loss_op)

                avg_kl_loss += np.mean(kl_loss)

                step = i * config.num_batches + b
                if step % config.print_step == 0:
                    logger.info('{}/{} (epoch {}) '
                                'loss: {:.3f}, gen loss: {:.3f}, '
                                'gen loss concept: {:.3f}, gen loss api: {:.3f}, '
                                'gen loss type: {:.3f}, gen loss clstype: {:.3f}, gen loss var: {:.3f}, '
                                'gen loss op: {:.3f}, gen loss method: {:.3f}, '
                                'KL loss: {:.3f}. '
                                .format(step,
                                        config.num_epochs * config.num_batches,
                                        i + 1, avg_loss / (b + 1), avg_ast_loss / (b + 1),
                                        avg_ast_gen_loss_concept / (b + 1), avg_ast_gen_loss_api / (b + 1),
                                        avg_gen_loss_type / (b + 1), avg_gen_loss_clstype / (b + 1),
                                        avg_ast_gen_loss_var / (b + 1),
                                        avg_ast_gen_loss_op / (b + 1), avg_ast_gen_loss_method / (b + 1),
                                        avg_kl_loss / (b + 1)))
                    logger.info('{}'.format([truncate_two_decimals(s) for s in sigma]))

            # pool.terminate()
            # logger.info("before close " + datetime.now().strftime("%H:%M:%S"))
            # pool.close()
            # logger.info("after close, before join " + datetime.now().strftime("%H:%M:%S"))
            # pool.join()
            # logger.info("after join " + datetime.now().strftime("%H:%M:%S"))
            total_memory, used_memory, free_memory = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
            memory_percent = round((used_memory/total_memory) * 100, 2)
            logger.info(str(total_memory) +'used_memory:'+ str(used_memory) +'free_memory:'+ str(free_memory) +'memory_percent:'+ str(memory_percent))
            if memory_percent > 88:
                sys.exit(0)

            if (i + 1) % config.checkpoint_step == 0:
                checkpoint_dir = os.path.join(clargs.save, 'model_decoder_recont{}.ckpt'.format(i + 1))
                saver.save(sess, checkpoint_dir)
                dump_json(read_json(os.path.join(clargs.data, 'compiler_data.json')),
                          os.path.join(clargs.save + '/compiler_data.json'))
                dump_json(dump_config(config), os.path.join(clargs.save + '/config.json'))
                dump_json({'pid': os.getpid()}, os.path.join(clargs.save + '/pid.json'))
                logger.info('Model checkpoint: {}. Average for epoch , '
                            'loss: {:.3f}'.format
                            (checkpoint_dir, avg_loss / config.num_batches))


# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--python_recursion_limit', type=int, default=10000,
                        help='set recursion limit for the Python interpreter')
    parser.add_argument('--save', type=str, default='save',
                        help='checkpoint model during training here')
    parser.add_argument('--data', type=str, default='../data_extraction/data_reader/data',
                        help='load data from here')
    parser.add_argument('--config', type=str, default=None,
                        help='config file (see description above for help)')
    parser.add_argument('--continue_from', type=str, default=None,
                        help='ignore config options and continue training model checkpointed here')
    clargs_ = parser.parse_args()
    if not os.path.exists(clargs_.save):
        os.makedirs(clargs_.save)
    sys.setrecursionlimit(clargs_.python_recursion_limit)
    if clargs_.config and clargs_.continue_from:
        parser.error('Do not provide --config if you are continuing from checkpointed model')
    if not clargs_.config and not clargs_.continue_from:
        parser.error('Provide at least one option: --config or --continue_from')
    train(clargs_)
