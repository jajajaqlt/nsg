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
import os

import argparse
import sys
import json

import tensorflow as tf
from data_extraction.data_reader.data_loader import Loader
from trainer_vae.model import Model
from trainer_vae.utils import read_config, dump_config
from utilities.basics import dump_json, truncate_two_decimals, read_json
from utilities.logging import create_logger

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_" \
                                  "BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# temp_labels, temp_nums = get_igmm_ins(apicalls[j], target_api[j], igmm_api_dict, 'api')
def add_igmm_ins(ev_labels, ev_nums, igmm_dict, prefix, ins_labels, ins_nums):
    # labels [10], nums [[10*256]]
    temp_labels = []
    temp_nums = []
    nonzero_count = np.count_nonzero(ev_labels)
    temp_labels = [prefix + igmm_dict[ev_labels[i]] for i in range(nonzero_count)]
    temp_nums = [ev_nums[i] for i in range(nonzero_count)]
    ins_labels.extend(temp_labels)
    ins_nums.extend(temp_nums)

def train(clargs):
    config_file = os.path.join(clargs.continue_from, 'config.json') \
        if clargs.continue_from is not None \
        else clargs.config
    with open(config_file) as f:
        config = read_config(json.load(f))

    loader = Loader(clargs.data, config)
    model = Model(config)

    # import pdb; pdb.set_trace()

    # adds api, kw and type dicts with indices as keys for igmm help function
    try:
        igmm_api_dict = {y: x for x, y in config.vocab.api_dict.items()}
        igmm_type_dict = {y: x for x, y in config.vocab.type_dict.items()}
        igmm_kw_dict = {y: x for x, y in config.vocab.kw_dict.items()}
    except:
        import pdb; pdb.set_trace()
    # import pdb; pdb.set_trace()

    logger = create_logger(os.path.join(clargs.save, 'loss_values.log'))
    logger.info('Process id is {}'.format(os.getpid()))
    logger.info('GPU device is {}'.format(os.environ["CUDA_VISIBLE_DEVICES"]))
    logger.info('Amount of data used is {}'.format(config.num_batches * config.batch_size))
    num_params = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
    logger.info('Number of params {}\n\t'.format(num_params))

    with tf.compat.v1.Session(
            config=tf.compat.v1.ConfigProto(log_device_placement=False,
                                            allow_soft_placement=True)) as sess:

        saver = tf.compat.v1.train.Saver(tf.global_variables(), max_to_keep=3)
        tf.global_variables_initializer().run()

        # restore model
        if clargs.continue_from is not None:
            vars_ = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            old_saver = tf.compat.v1.train.Saver(vars_)
            ckpt = tf.train.get_checkpoint_state(clargs.continue_from)
            old_saver.restore(sess, ckpt.model_checkpoint_path)

        # training
        for i in range(config.num_epochs):
            loader.reset_batches()
            avg_loss, avg_ast_loss, avg_kl_loss = 0., 0., 0.,
            avg_ast_gen_loss_concept, avg_ast_gen_loss_api, \
            avg_gen_loss_type, avg_gen_loss_clstype, avg_ast_gen_loss_var, \
            avg_ast_gen_loss_vardecl, avg_ast_gen_loss_method = 0., 0., 0., 0., 0., 0., 0.

            for b in range(config.num_batches):
                nodes, edges, targets, var_decl_ids, ret_reached,\
                node_type_number, \
                type_helper_val, expr_type_val, ret_type_val, \
                all_var_mappers, iattrib, \
                ret_type, fp_in, fields, \
                apicalls, types, keywords, method, classname, javadoc_kws,\
                    surr_ret, surr_fp, surr_method = loader.next_batch()
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

                # run the optimizer
                loss, ast_loss, \
                ast_gen_loss_concept, ast_gen_loss_api, \
                ast_gen_loss_type, ast_gen_loss_clstype, ast_gen_loss_var, \
                ast_gen_loss_vardecl, ast_gen_loss_method, \
                kl_loss, _, sigma = \
                    sess.run([model.loss, model.ast_gen_loss,
                              model.ast_gen_loss_concept, model.ast_gen_loss_api,
                              model.ast_gen_loss_type, model.ast_gen_loss_clstype,
                              model.ast_gen_loss_var, model.ast_gen_loss_vardecl, model.ast_gen_loss_method,
                              model.KL_loss, model.train_op, model.encoder.program_encoder.sigmas], feed_dict=feed_dict)

                avg_loss += np.mean(loss)
                avg_ast_loss += np.mean(ast_loss)

                avg_ast_gen_loss_concept += np.mean(ast_gen_loss_concept)
                avg_ast_gen_loss_method += np.mean(ast_gen_loss_method)
                avg_ast_gen_loss_api += np.mean(ast_gen_loss_api)
                avg_gen_loss_type += np.mean(ast_gen_loss_type)
                avg_gen_loss_clstype += np.mean(ast_gen_loss_clstype)
                avg_ast_gen_loss_var += np.mean(ast_gen_loss_var)
                avg_ast_gen_loss_vardecl += np.mean(ast_gen_loss_vardecl)

                avg_kl_loss += np.mean(kl_loss)

                if i == 99:
                    print('writing vectors')
                    # adds code to store all vectors
                    # also writes apicalls, types, keywords
                    # import pdb; pdb.set_trace()
                    # everything is 128*10*256, except the ret 128*256, datatypes all float
                    target_api, target_types, target_kws, target_fp, \
                    target_field, target_ret, target_method, target_class, \
                    target_jd, target_surr = sess.run(
                        [model.encoder.program_encoder.ast_mean_api.reshaper,
                         model.encoder.program_encoder.ast_mean_types.reshaper,
                         model.encoder.program_encoder.ast_mean_kws.reshaper,
                         model.encoder.program_encoder.fp_mean_enc.reshaper,
                         model.encoder.program_encoder.field_enc.reshaper,
                         model.encoder.program_encoder.ret_mean_enc.latent_encoding,
                         model.encoder.program_encoder.method_enc.reshaper,
                         model.encoder.program_encoder.class_enc.reshaper,
                         model.encoder.program_encoder.jd_enc.reshaper,
                         model.encoder.program_encoder.surr_enc.layer3],
                        feed_dict=feed_dict)

                    # save non-zero output data into data_ls list
                    # format [[[labels],[numbers]],...]
                    data_ls = []
                    data_ls_name = 'igmm_input_batch_' + str(i) + '_' + str(b)
                    for j in range(config.batch_size):
                        ins_labels = []
                        ins_nums = []
                        # 0. api
                        add_igmm_ins(apicalls[j], target_api[j], igmm_api_dict, 'api_', ins_labels, ins_nums)
                        # 1. types
                        add_igmm_ins(types[j], target_types[j], igmm_type_dict, 'types_', ins_labels, ins_nums)
                        # 2. kws
                        add_igmm_ins(keywords[j], target_kws[j], igmm_kw_dict, 'kws_', ins_labels, ins_nums)
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
                        # surr_method change from 128 10 3 int to 128 10 1 int
                        surr_method_first = [m[0] for m in surr_method[j]]
                        add_igmm_ins(surr_method_first, target_surr[j], igmm_kw_dict, 'surr_', ins_labels, ins_nums)
                        data_ls.append([ins_labels, ins_nums])
                    # dataypes all int
                    # 0
                    # model.apicalls: apicalls 128 10
                    # 1
                    # model.types: types 128 10
                    # 2
                    # model.keywords: keywords 128 10
                    # 3
                    # model.formal_param_inputs: fp_in 128 10
                    # 4
                    # model.field_inputs: fields 128 10
                    # 5
                    # model.return_type: ret_type 128
                    # 6
                    # model.method: method 128 3
                    # 7
                    # model.classname: classname 128 3
                    # 8
                    # model.javadoc_kws: javadoc_kws 128 10
                    # 9
                    # model.surr_method: surr_method 128 10 3
                    # import pdb; pdb.set_trace()
                    np.save(data_ls_name, data_ls, allow_pickle=True)

                    # import pdb; pdb.set_trace()

                step = i * config.num_batches + b
                if step % config.print_step == 0:
                    logger.info('{}/{} (epoch {}) '
                                'loss: {:.3f}, gen loss: {:.3f}, '
                                'gen loss concept: {:.3f}, gen loss api: {:.3f}, '
                                'gen loss type: {:.3f}, gen loss clstype: {:.3f}, gen loss var: {:.3f}, '
                                'gen loss vardecl: {:.3f}, gen loss method: {:.3f}, '
                                'KL loss: {:.3f}. '
                                .format(step,
                                        config.num_epochs * config.num_batches,
                                        i + 1, avg_loss / (b + 1), avg_ast_loss / (b + 1),
                                        avg_ast_gen_loss_concept / (b + 1), avg_ast_gen_loss_api / (b + 1),
                                        avg_gen_loss_type / (b + 1), avg_gen_loss_clstype / (b + 1),
                                        avg_ast_gen_loss_var / (b + 1),
                                        avg_ast_gen_loss_vardecl / (b + 1), avg_ast_gen_loss_method / (b + 1),
                                        avg_kl_loss / (b + 1)))
                    logger.info('{}'.format([truncate_two_decimals(s) for s in sigma]))

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
