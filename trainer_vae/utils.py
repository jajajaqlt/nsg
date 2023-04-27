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
from multiprocessing import Pool
from operator import itemgetter
import argparse

from data_extraction.data_reader.utils import read_vocab, dump_vocab
from trainer_vae.igmm_np import multi_d_igmm, lam, rha, beta, omega, alpha
import numpy as np


CONFIG_GENERAL = ['batch_size', 'num_epochs', 'latent_size',
                  'ev_drop_rate', "ev_miss_rate", "decoder_drop_rate",
                  'learning_rate', 'max_ast_depth', 'input_fp_depth',
                  'max_keywords', 'max_variables', 'max_fields', 'max_camel_case',
                  'trunct_num_batch', 'print_step', 'checkpoint_step', 'max_means',]
CONFIG_ENCODER = ['units', 'num_layers']
CONFIG_DECODER = ['units', 'num_layers']



# convert JSON to config
def read_config(js, infer=False):
    config = argparse.Namespace()

    for attr in CONFIG_GENERAL:
        config.__setattr__(attr, js[attr])

    config.encoder = argparse.Namespace()
    for attr in CONFIG_ENCODER:
        config.encoder.__setattr__(attr, js['encoder'][attr])

    config.decoder = argparse.Namespace()
    for attr in CONFIG_DECODER:
        config.decoder.__setattr__(attr, js['decoder'][attr])

    if infer:
        config.vocab = read_vocab(js['vocab'])

    return config


# convert config to JSON
def dump_config(config):
    js = {}

    for attr in CONFIG_GENERAL:
        js[attr] = config.__getattribute__(attr)

    js['encoder'] = {attr: config.encoder.__getattribute__(attr) for attr in
                     CONFIG_ENCODER}

    js['decoder'] = {attr: config.decoder.__getattribute__(attr) for attr in
                     CONFIG_DECODER}

    js['vocab'] = dump_vocab(config.vocab)

    return js


# temp_labels, temp_nums = get_igmm_ins(apicalls[j], target_api[j], igmm_api_dict, 'api')
# add_igmm_ins([surr_ret[j], surr_method[j], surr_fp[j]], target_surr[j], [igmm_type_dict, igmm_kw_dict], 'surr_', ins_labels, ins_nums)
def add_igmm_ins(ev_labels, ev_nums, igmm_dict, prefix, ins_labels, ins_nums):
    # labels [10], nums [[10*256]]
    temp_labels = []
    temp_nums = []

    if prefix == 'surr_':
        # ev_labels[0] = surr_ret[j]
        # max_keywords
        rets = ev_labels[0]
        # max_keywords * max_camel_case
        methodnames = ev_labels[1]
        # max_keywords * max_camel_case
        fps = ev_labels[2]

        type_dict = igmm_dict[0]
        kw_dict = igmm_dict[1]

        nonzero_count = np.count_nonzero(rets)
        # make header labels
        temp_labels = []
        for i in range(nonzero_count):
            temp_label = prefix + type_dict[rets[i]]+ '::'
            # method names
            nonzero = np.count_nonzero(methodnames[i])
            temp_label += '_'.join([kw_dict[mn] for mn in methodnames[i][: nonzero]]) + '('
            # formal parameters
            nonzero = np.count_nonzero(fps[i])
            temp_label += ','.join([type_dict[fp] for fp in fps[i][: nonzero]]) + ')'
            temp_labels.append(temp_label)
    else:
        nonzero_count = np.count_nonzero(ev_labels)
        temp_labels = [prefix + igmm_dict[ev_labels[i]] for i in range(nonzero_count)]

    temp_nums = [ev_nums[i] for i in range(nonzero_count)]
    # adds indices to labels
    for i in range(len(temp_labels)):
        temp_labels[i] = str(i) + '_' + temp_labels[i]
    ins_labels.extend(temp_labels)
    ins_nums.extend(temp_nums)


def run_igmm_and_fill_feed(sess, encoder, config, feed, 
                           fp_in, fields, ret_type, method, classname, 
                           javadoc_kws, surr_ret, surr_method, surr_fp):
    # extract evidences // normal way
    print('extracting vectors')
    # everything is 128*10*256, except the ret 128*256, datatypes all float
    target_fp, target_field, target_ret, target_method, target_class, \
    target_jd, target_surr = sess.run(
        [encoder.program_encoder.fp_mean_enc.reshaper,
            encoder.program_encoder.field_enc.reshaper,
            encoder.program_encoder.ret_mean_enc.latent_encoding,
            encoder.program_encoder.method_enc.reshaper,
            encoder.program_encoder.class_enc.reshaper,
            encoder.program_encoder.jd_enc.reshaper,
            encoder.program_encoder.surr_enc.layer3],
        feed_dict=feed)

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

    # adds api, kw and type dicts with indices as keys for igmm help function
    try:
        igmm_api_dict = {y: x for x, y in config.vocab.api_dict.items()}
        igmm_type_dict = {y: x for x, y in config.vocab.type_dict.items()}
        igmm_kw_dict = {y: x for x, y in config.vocab.kw_dict.items()}
    except:
        import pdb; pdb.set_trace()

    batch_igmm_input = []
    batch_igmm_params = []
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
        str_idx = 'testcase_' + str(j)
        igmm_samples = 50
        prev_indicators, prev_means, prev_precs = None, None, None
        batch_igmm_params.append([str_idx, config.latent_size, ins_nums, igmm_samples, prev_indicators, prev_means, prev_precs])

    # runs igmm, compute indicator matrices and fills placeholders
    pool = Pool(processes=40)
    num_chunks = config.batch_size
    for result in pool.map(multi_d_igmm, batch_igmm_params[:num_chunks]):
        str_idx, indicators, means, precs = result
        # strings
        _, j = str_idx.split('_')
        j = int(j)
        ins_labels = batch_igmm_input[j][0]
        pair_results = list(zip(indicators, ins_labels))

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
        print(str_idx)
        print(*sorted_cluster, sep='\n')

    # feed_dict.update({model.latent_vectors: latent_vectors})
    feed.update({encoder.program_encoder.lamda: lam,
                        encoder.program_encoder.rha: rha})
    # feed precions, counts and indicators
    keys_and_values_pair = [(encoder.program_encoder.precisions, precisions),
                            (encoder.program_encoder.counts, counts),
                            (encoder.program_encoder.fp_mean_enc_indmats, fp_inds),
                            (encoder.program_encoder.field_enc_indmats, field_inds),
                            (encoder.program_encoder.ret_mean_enc_indvecs, ret_inds),
                            (encoder.program_encoder.method_enc_indmats, method_inds),
                            (encoder.program_encoder.class_enc_indmats, class_inds),
                            (encoder.program_encoder.jd_enc_indmats, jd_inds),
                            (encoder.program_encoder.surr_enc_indmats, surr_inds)]
    for (keys, values) in keys_and_values_pair:
        dictionary = dict(zip(keys, values))
        feed.update(dictionary)    