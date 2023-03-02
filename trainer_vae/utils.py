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
import argparse
from data_extraction.data_reader.utils import read_vocab, dump_vocab
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
