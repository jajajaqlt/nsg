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

from program_helper.program_reader import ProgramReader
import os
import numpy as np


class Loader:
    def __init__(self, data_path, config):
        self.config = config

        self.program_reader = ProgramReader(
             max_ast_depth=config.max_ast_depth,
             max_fp_depth=config.input_fp_depth,
             max_keywords=config.max_keywords,
             data_path=data_path
        )

        self.program_reader.load_data()
        self.config.vocab = self.program_reader.vocab

        max_num_batches = int( self.program_reader.get_size() / self.config.batch_size)
        print('Max batches :: {}'.format(max_num_batches))
        if self.config.trunct_num_batch is None:
            self.config.num_batches = max_num_batches
        else:
            self.config.num_batches = min(self.config.trunct_num_batch, max_num_batches)
        assert self.config.num_batches > 0, 'Not enough data'
        sz = self.config.num_batches * self.config.batch_size

        self.program_reader.truncate(sz)
        self.program_reader.split(self.config.num_batches)

        self.nodes, self.edges, self.targets, \
            self.var_decl_ids, self.return_reached,\
            self.node_type_number, \
            self.type_helper_val, self.expr_type_val, self.ret_type_val, \
                self.all_var_mappers, self.iattrib \
                    = self.program_reader.ast_reader.get()
        self.return_types = self.program_reader.return_type_reader.get()
        self.fp_input = self.program_reader.formal_param_reader.get()
        self.apicalls, self.types, self.keywords,\
            self.method, self.classname, self.javadoc_kws = self.program_reader.keyword_reader.get()

        self.field_inputs = self.program_reader.field_reader.get()
        self.surr_ret_types, self.surr_fp_types, self.surr_methods = self.program_reader.surrounding_reader.get()

        # adds the latent_matrices here
        print('starts reading latent matrices here!')
        self.read_latent_vectors(data_path, config, sz, self.config.num_batches)

        self.reset_batches()
        print('Done')


    def read_latent_vectors(self, data_path, config, sz, num_batches):
        # reads the latent matrices npy file
        # filename = '110522ev_vec_means.npy'
        # filename = '/home/lq4/other_projects/gibbs_sampler/010423big_ev_vec_means.npy'
        # filename = '2dp_means.npy'
        filename = '/home/letao/gibbs_sampler/1dp3means+1.npy'
        file_path = os.path.join(data_path, filename)
        with open(file_path, 'rb') as f:
            raw_vectors = np.load(f, allow_pickle=True)
        # wrangles the data a little bit, raw_matrices is [data_length, num_gauss, encoder.units]
        data_length = len(raw_vectors)
        self.latent_vectors = np.zeros((data_length, config.max_means, config.encoder.units), dtype=np.float32)
        for i, m in enumerate(raw_vectors):
            len_list = min(len(m), config.max_means)
            mod_list = m[: len_list]
            self.latent_vectors[i, :len_list] = mod_list
        # truncates the data 
        self.latent_vectors = self.latent_vectors[:sz, :config.max_means]
        # splits the data
        self.latent_vectors = np.split(self.latent_vectors, num_batches, axis=0)
        print('finishes reading the latent vectors')
        return



    def reset_batches(self):
        self.batches = iter(
            zip(self.nodes, self.edges, self.targets,
                self.var_decl_ids, self.return_reached,
                self.node_type_number,
                self.type_helper_val, self.expr_type_val, self.ret_type_val,
                self.all_var_mappers,
                self.iattrib,
                self.return_types,
                self.fp_input,
                self.field_inputs,
                self.apicalls, self.types, self.keywords, self.method, self.classname, self.javadoc_kws,
                self.surr_ret_types, self.surr_fp_types, self.surr_methods, self.latent_vectors))
        return

    def next_batch(self):
        n, e, t, \
        var_decls, ret_reached, \
        node_type_number,\
        thv, etv, ncv, \
        all_var_mappers, \
        iattrib, \
        rt, \
        fp_in, \
        fields,\
        apis, types, kws, mn, cn, jkw, s_rt, s_fp, s_m, l_m = next(self.batches)
        return n, e, t, \
               var_decls, ret_reached, \
               node_type_number,\
               thv, etv, ncv, \
               all_var_mappers, \
               iattrib, \
               rt, \
               fp_in,\
               fields,\
               apis, types, kws, mn, cn, jkw, \
               s_rt, s_fp, s_m, l_m
