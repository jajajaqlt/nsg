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
import tensorflow as tf

from program_helper.element.dense_encoder import DenseEncoder
from program_helper.sequence.sequence_encoder import SequenceEncoder
from program_helper.set.set_encoder import KeywordEncoder
from program_helper.surrounding.surrounding_encoder import SurroundingEncoder


class ProgramEncoder:

    def __init__(self, config,
                 apicalls, types, keywords,
                 fp_inputs, field_inputs,
                 ret_type,
                 method, classname, javadoc,
                 surr_ret, surr_fp, surr_method,
                 ev_drop_rate=None,
                 ev_miss_rate=None
                 ):

        self.type_emb = tf.get_variable('emb_type', [config.vocab.type_dict_size, config.encoder.units])

        self.apiname_emb = tf.get_variable('emb_apiname', [config.vocab.api_dict_size, config.encoder.units])
        self.typename_emb = tf.get_variable('emb_typename', [config.vocab.type_dict_size, config.encoder.units])
        self.kw_emb = tf.get_variable('emb_kw', [config.vocab.kw_dict_size, config.encoder.units])

        '''
            let us unstack the drop prob 
        '''
        ev_drop_rate = tf.unstack(ev_drop_rate)
        with tf.variable_scope("ast_tree_api"):
            self.ast_mean_api = KeywordEncoder(apicalls,
                                               config.encoder.units, config.encoder.num_layers,
                                               config.vocab.apiname_dict_size,
                                               config.batch_size,
                                               config.max_keywords,
                                               emb=self.apiname_emb,
                                               drop_prob=ev_drop_rate[0]
                                               )
            ast_mean_api = self.ast_mean_api.output
            #not applicable for sub batch key encoder

        with tf.variable_scope("ast_tree_types"):
            self.ast_mean_types = KeywordEncoder(types,
                                                 config.encoder.units, config.encoder.num_layers,
                                                 config.vocab.typename_dict_size,
                                                 config.batch_size,
                                                 config.max_keywords,
                                                 emb=self.typename_emb,
                                                 drop_prob=ev_drop_rate[1]
                                                 )
            ast_mean_types = self.ast_mean_types.output
            #not applicable for sub batch key encoder

        with tf.variable_scope("ast_tree_kw"):
            self.ast_mean_kws = KeywordEncoder(keywords,
                                               config.encoder.units, config.encoder.num_layers,
                                               config.vocab.kw_dict_size,
                                               config.batch_size,
                                               config.max_keywords,
                                               emb=self.kw_emb,
                                               drop_prob=ev_drop_rate[2]
                                               )
            ast_mean_kw = self.ast_mean_kws.output
            #not applicable for sub batch key encoder

        # all types of evidences each having max_means ind_mats except return_type (max_means ind_vecs)
        with tf.variable_scope("formal_param"):
            self.fp_mean_enc = KeywordEncoder( fp_inputs,
                                               config.encoder.units, config.encoder.num_layers,
                                               config.vocab.type_dict_size,
                                               config.batch_size,
                                               config.input_fp_depth,
                                               emb=self.type_emb,
                                               drop_prob=ev_drop_rate[3],
                                               layers_name='fp_mean_enc'
                                               )
            fp_mean = self.fp_mean_enc.output
            # config.max_means
            # max_means many indicator matrices placeholders
            self.fp_mean_enc_splits = []
            self.fp_mean_enc_indmats = []
            for i in range(config.max_means):
                fp_mean_enc_indmat = tf.placeholder(tf.int32, [config.batch_size, config.input_fp_depth])
                self.fp_mean_enc_indmats.append(fp_mean_enc_indmat)
                fp_mean_enc_split = KeywordEncoder(fp_inputs,
                                                config.encoder.units, config.encoder.num_layers,
                                                config.vocab.type_dict_size,
                                                config.batch_size,
                                                config.input_fp_depth,
                                                emb=self.type_emb,
                                                drop_prob=ev_drop_rate[3],
                                                indicator_matrix=fp_mean_enc_indmat,                                                
                                                layers_name='fp_mean_enc',
                                                layers_reuse=True
                                                )
                self.fp_mean_enc_splits.append(fp_mean_enc_split)

        with tf.variable_scope("field_vars"):
            self.field_enc = KeywordEncoder(field_inputs,
                                             config.encoder.units, config.encoder.num_layers,
                                             config.vocab.type_dict_size,
                                             config.batch_size,
                                             config.max_fields,
                                             emb=self.type_emb,
                                             drop_prob=ev_drop_rate[4],
                                             layers_name='field_enc'
                                            )
            field_mean = self.field_enc.output
            # config.max_means
            self.field_enc_splits = []
            self.field_enc_indmats = []
            for i in range(config.max_means):
                field_enc_indmat = tf.placeholder(tf.int32, [config.batch_size, config.max_fields])
                self.field_enc_indmats.append(field_enc_indmat)
                field_enc_split = KeywordEncoder(field_inputs,
                                                config.encoder.units, config.encoder.num_layers,
                                                config.vocab.type_dict_size,
                                                config.batch_size,
                                                config.max_fields,
                                                emb=self.type_emb,
                                                drop_prob=ev_drop_rate[4],
                                                indicator_matrix=field_enc_indmat,
                                                layers_name='field_enc',
                                                layers_reuse=True
                                                )
                self.field_enc_splits.append(field_enc_split)

        with tf.variable_scope("ret_type", reuse=tf.AUTO_REUSE):
            self.ret_mean_enc = DenseEncoder(ret_type,
                                             config.encoder.units, config.encoder.num_layers,
                                             config.latent_size,
                                             config.vocab.type_dict_size, config.batch_size,
                                             emb=self.type_emb,
                                             drop_prob=ev_drop_rate[5],
                                             layers_name='ret_mean_enc',
                                             )
            ret_mean = self.ret_mean_enc.latent_encoding
            # config.max_means
            self.ret_mean_enc_splits = []
            self.ret_mean_enc_indvecs = []
            for i in range(config.max_means):
                ret_mean_enc_indvec = tf.placeholder(tf.int32, [config.batch_size])
                self.ret_mean_enc_indvecs.append(ret_mean_enc_indvec)
                ret_mean_enc_split = DenseEncoder(ret_type,
                                                config.encoder.units, config.encoder.num_layers,
                                                config.latent_size,
                                                config.vocab.type_dict_size, config.batch_size,
                                                emb=self.type_emb,
                                                drop_prob=ev_drop_rate[5],
                                                indicator_vec=ret_mean_enc_indvec,
                                                layers_name='ret_mean_enc',
                                                layers_reuse=True
                                                )
                self.ret_mean_enc_splits.append(ret_mean_enc_split)

        with tf.variable_scope("method_kw"):
            self.method_enc = KeywordEncoder(method,
                                             config.encoder.units, config.encoder.num_layers,
                                             config.vocab.kw_dict_size,
                                             config.batch_size,
                                             config.max_camel_case,
                                             emb=self.kw_emb,
                                             drop_prob=ev_drop_rate[6],
                                             layers_name='method_enc'
                                             )
            method_mean_kw = self.method_enc.output
            # config.max_means
            self.method_enc_splits = []
            self.method_enc_indmats = []
            for i in range(config.max_means):
                method_enc_indmat = tf.placeholder(tf.int32, [config.batch_size, config.max_camel_case])
                self.method_enc_indmats.append(method_enc_indmat)
                method_enc_split = KeywordEncoder(method,
                                                config.encoder.units, config.encoder.num_layers,
                                                config.vocab.kw_dict_size,
                                                config.batch_size,
                                                config.max_camel_case,
                                                emb=self.kw_emb,
                                                drop_prob=ev_drop_rate[6],
                                                indicator_matrix=method_enc_indmat,
                                                layers_name='method_enc',
                                                layers_reuse=True
                                                )
                self.method_enc_splits.append(method_enc_split)

        with tf.variable_scope("class_kw"):
            self.class_enc = KeywordEncoder(classname,
                                            config.encoder.units, config.encoder.num_layers,
                                            config.vocab.kw_dict_size,
                                            config.batch_size,
                                            config.max_camel_case,
                                            emb=self.kw_emb,
                                            drop_prob=ev_drop_rate[7],
                                            layers_name='class_enc'
                                            )
            class_mean_kw = self.class_enc.output
            # config.max_means
            self.class_enc_splits = []
            self.class_enc_indmats = []
            for i in range(config.max_means):
                class_enc_indmat = tf.placeholder(tf.int32, [config.batch_size, config.max_camel_case])
                self.class_enc_indmats.append(class_enc_indmat)
                class_enc_split = KeywordEncoder(classname,
                                                config.encoder.units, config.encoder.num_layers,
                                                config.vocab.kw_dict_size,
                                                config.batch_size,
                                                config.max_camel_case,
                                                emb=self.kw_emb,
                                                drop_prob=ev_drop_rate[7],
                                                indicator_matrix=class_enc_indmat,
                                                layers_name='class_enc',
                                                layers_reuse=True
                                                )
                self.class_enc_splits.append(class_enc_split)

        with tf.variable_scope("javadoc_kw"):
            self.jd_enc = KeywordEncoder(javadoc,
                                         config.encoder.units, config.encoder.num_layers,
                                         config.vocab.kw_dict_size,
                                         config.batch_size,
                                         config.max_keywords,
                                         emb=self.kw_emb,
                                         drop_prob=ev_drop_rate[8],
                                         layers_name='jd_enc'
                                         )
            jd_mean = self.jd_enc.output
            # config.max_means
            self.jd_enc_splits = []
            self.jd_enc_indmats = []
            for i in range(config.max_means):
                jd_enc_indmat = tf.placeholder(tf.int32, [config.batch_size, config.max_keywords])
                self.jd_enc_indmats.append(jd_enc_indmat)
                jd_enc_split = KeywordEncoder(javadoc,
                                            config.encoder.units, config.encoder.num_layers,
                                            config.vocab.kw_dict_size,
                                            config.batch_size,
                                            config.max_keywords,
                                            emb=self.kw_emb,
                                            drop_prob=ev_drop_rate[8],
                                            indicator_matrix=jd_enc_indmat,
                                            layers_name='jd_enc',
                                            layers_reuse=True
                                            )
                self.jd_enc_splits.append(jd_enc_split)

        with tf.variable_scope("surrounding"):
            self.surr_enc = SurroundingEncoder(surr_ret, surr_fp, surr_method,
                                         config.encoder.units, config.encoder.num_layers,
                                         config.vocab.type_dict_size,
                                         config.vocab.kw_dict_size,
                                         config.batch_size,
                                         config.max_keywords,
                                         config.max_camel_case,
                                         self.type_emb,
                                         self.kw_emb,
                                         drop_prob=ev_drop_rate[9],
                                         layers_name='surr_enc'
                                         )
            surr_mean = self.surr_enc.output
            # config.max_means
            self.surr_enc_splits = []
            self.surr_enc_indmats = []
            for i in range(config.max_means):
                # correct b/c ret, fp, method all belongs to one method or not
                surr_enc_indmat = tf.placeholder(tf.int32, [config.batch_size, config.max_keywords])
                self.surr_enc_indmats.append(surr_enc_indmat)
                surr_enc_split = SurroundingEncoder(surr_ret, surr_fp, surr_method,
                                                config.encoder.units, config.encoder.num_layers,
                                                config.vocab.type_dict_size,
                                                config.vocab.kw_dict_size,
                                                config.batch_size,
                                                config.max_keywords,
                                                config.max_camel_case,
                                                self.type_emb,
                                                self.kw_emb,
                                                drop_prob=ev_drop_rate[9],
                                                indicator_matrix=surr_enc_indmat,
                                                layers_name='surr_enc',
                                                layers_reuse=True
                                                )
                self.surr_enc_splits.append(surr_enc_split) 

        evidences = [ast_mean_api, ast_mean_types, ast_mean_kw,
                     fp_mean, ret_mean,
                     field_mean,
                     method_mean_kw, class_mean_kw, jd_mean, surr_mean,
                     ]

        # foreach mean, sums up all outputs of sub batch encoders
        # placeholders for precisions, lamdas and rhas
        # reference rohan's impl

        # d = 1. + tf.reduce_sum(tf.stack(d), axis=0)
        # lambda, rha single or multi-dim? check how ev_drop manipulated
        # lamda and rhas stay the same for all means
        self.lamda = tf.placeholder(tf.float32, shape=())
        self.rha = tf.placeholder(tf.float32, shape=())
        self.precisions = []
        # check how rohan computes this
        self.counts = []
        for i in range(config.max_means):
            self.precisions.append(tf.placeholder(tf.float32, [config.batch_size, config.latent_size]))
            self.counts.append(tf.placeholder(tf.float32, [config.batch_size]))

        # WIP
        latent_states = []
        for i in range(config.max_means):
            # [(bs, units)+num_ev]
            split_evidences = [self.fp_mean_enc_splits[i].output, self.field_enc_splits[i].output, 
                               self.ret_mean_enc_splits[i].latent_encoding, self.method_enc_splits[i].output,
                               self.class_enc_splits[i].output, self.jd_enc_splits[i].output,
                               self.surr_enc_splits[i].output]
            split_evidences_sum = tf.reduce_sum(tf.stack(split_evidences), axis=0)
            mean = (split_evidences_sum * self.precisions[i] + self.lamda * self.rha) / (
                tf.reshape(self.counts[i], [-1, 1]) * self.precisions[i] + self.rha)
            covar = 1 / (tf.reshape(self.counts[i], [-1, 1]) * self.precisions[i] + self.rha)
            # import pdb; pdb.set_trace()
            # starts here, check how rohan did
            # normal_mean = (tf.math.add_n(split_evidences) * self.precisions[i] + self.lamda * self.rha) / (self.counts[i] * self.precisions[i] + self.rha)
            # checks variances vv stddev
            samples = tf.random.normal([config.batch_size, config.latent_size], mean=0., stddev=1.,
                                       dtype=tf.float32)
            # self.latent_state = self.encoder.output_mean + tf.sqrt(self.encoder.output_covar) * samples
            latent_state = mean + tf.sqrt(covar) * samples
            latent_states.append(latent_state)           

        self.latent_vectors = tf.stack(latent_states, axis=1)
 
        '''
        Lets drop some evidence types altogether according to #ev_miss_rate
        '''
        # import pdb; pdb.set_trace()
        # [(bs, units)+num_ev]
        evidences = [tf.where(tf.random_uniform((config.batch_size, config.latent_size), 0, 1, dtype=tf.float32) > ev_miss_rate[j],
                              ev,
                              tf.zeros_like(ev)) for j, ev in enumerate(evidences)]


        with tf.variable_scope('sigma'):
            sigmas = tf.get_variable('sigma', [len(evidences)])
            # [()+num_ev]
            sigmas = tf.unstack(sigmas)

        # [(batch_size)+num_ev]
        # checks which ev has no evidence
        d = [tf.where(tf.not_equal(tf.reduce_sum(ev, axis=1), 0.),
                      tf.tile([1. / tf.square(sigma)], [config.batch_size]),
                      tf.zeros(config.batch_size)) for ev, sigma in zip(evidences, sigmas)]
        # (bs) // n = nsigma+...
        d = 1. + tf.reduce_sum(tf.stack(d), axis=0)
        # (bs, units)
        denom = tf.tile(tf.reshape(d, [-1, 1]), [1, config.latent_size])

        # [(bs, units)+num_ev]
        encodings = [ev / tf.square(sigma) for ev, sigma in
                     zip(evidences, sigmas)]
        # [(bs, units)+num_ev] // for reducing the number of params?
        encodings = [tf.where(tf.not_equal(tf.reduce_sum(enc, axis=1), 0.),
                                   enc,
                                   tf.zeros([config.batch_size, config.latent_size], dtype=tf.float32)
                                   ) for enc in encodings]


        # same as above
        self.sigmas = sigmas
        # (bs, units)
        self.mean = tf.reduce_sum(tf.stack(encodings, axis=0), axis=0) / denom
        I = tf.ones([config.batch_size, config.latent_size], dtype=tf.float32)
        self.covar = I / denom

