import numpy as np
#from matplotlib import pyplot as plt
from scipy.stats import rv_discrete
from BigNumber.BigNumber import BigNumber
from BigNumber.BigNumber import log_base
from scipy.special import logsumexp
import math

# dims are independent, define 1-var normal and gamma parameters used for all dims
# lam = 0
# rha = 10
# beta = 1
# omega = 1 # 1
# alpha = 1

lam = 0
rha = 1000
beta = 1
omega = 0.001 # 1
alpha = 1

# total_testing_samples: batch * dim
def multi_d_igmm(args):
    str_idx, testing_dimensions, total_testing_samples, num_draws, prev_indicators, prev_means, prev_precs = args
    print('working on dp {} now'.format(str_idx))
    # plt.show()

    # from now on, applies gibbs sampling on gmm and draw samples
    # params for sampling
    burn_in = 0
    sample_draws = num_draws # 1000

    # hyper-parameters
    # original = True
    # if original:
    #     lam = 0
    #     rha = 1000
    #     beta = 1
    #     omega = 0.001
    #     alpha = 1
    debug = False

    # above is re-usable

    # # initial means and precisions
    # all_means = np.empty([testing_mixtures, testing_dimensions])
    # all_precs = np.empty([testing_mixtures, testing_dimensions])
    # for i in range(testing_mixtures):
    #     for j in range(testing_dimensions):
    #         all_means[i][j] = np.random.normal(lam, np.sqrt(rha ** -1))
    #         all_precs[i][j] = np.random.gamma(beta, omega ** -1)
    # print('Initial means are: ')
    # print(all_means)
    # print('Initial precisions are: ')
    # print(all_precs)
    #
    # # randomly assign membership to data points
    # total_ex_samples = total_testing_samples
    # random_labels = np.random.randint(testing_mixtures, size=[np.shape(total_ex_samples)[0], 1])
    # total_ex_samples = np.append(random_labels, total_ex_samples, axis=1)
    # print('Random assignments initialization are: ')
    # print(total_ex_samples)

    # initialization
    # format [['label', x, y], ...]
    total_ex_samples = total_testing_samples
    total_samples_count = np.shape(total_ex_samples)[0]
    data = []
    for i in range(total_samples_count):
        datapoint = []
        for j in range(testing_dimensions):
            datapoint.append(total_ex_samples[i][j])
        # data.append([total_ex_samples[i][1], total_ex_samples[i][2], total_ex_samples[i][3]])
        data.append(datapoint)
    #print('The data to use is: ')
    #print(data)
    num_data_points = len(data)
    if prev_indicators == None:
        num_classes = 1
        num_points_in_classes = [total_samples_count]
    else:
        num_classes = len(set(prev_indicators))
        num_points_in_classes = []
        for i in range(num_classes):
            num_points_in_classes.append(prev_indicators.count(i))

    # from here until the end of indicators, eliminate x, y, z,...

    # init_mean_x = np.random.normal(lam, np.sqrt(rha ** -1))
    # init_mean_y = np.random.normal(lam, np.sqrt(rha ** -1))
    # means = [[init_mean_x, init_mean_y]]
    if prev_means == None:
        init_means = np.empty([testing_dimensions])
        for i in range(testing_dimensions):
            init_means[i] = np.random.normal(lam, np.sqrt(rha ** -1))
        means = [init_means]
    else:
        means = prev_means

    # init_prec_x = np.random.gamma(beta, omega ** -1)
    # init_prec_y = np.random.gamma(beta, omega ** -1)
    # precs = [[init_prec_x, init_prec_y]]
    if prev_precs == None:
        init_precs = np.empty([testing_dimensions])
        for i in range(testing_dimensions):
            init_precs[i] = np.random.gamma(beta, omega ** -1)
        # import pdb; pdb.set_trace()
        precs = [init_precs]
    else:
        precs = prev_precs
    
    if prev_indicators == None:
        init_indicators = [0 for i in range(num_data_points)]
        indicators = init_indicators
    else:
        indicators = prev_indicators
    # finish initialization

    # total_samples_count = np.sum(testing_samples_count)
    for i in range(burn_in + sample_draws):
        if (i + 1) % 10 == 0 or i == 0:
            print(str_idx)
            print(i + 1)
            print(indicators)

        # update means

        # precompute y_avg
        y_total = [[0 for k in range(testing_dimensions)] for j in range(num_classes)]
        for j in range(num_data_points):
            # x, y = data[j]
            indicator = indicators[j]
            # multi-d
            # y_total[indicator][0] += x
            # y_total[indicator][1] += y
            for k in range(testing_dimensions):
                y_total[indicator][k] += data[j][k]

        for j in range(num_classes):
            for k in range(testing_dimensions):
                norm_mean = (y_total[j][k] * precs[j][k] + lam * rha) / (num_points_in_classes[j] * precs[j][k] + rha)
                norm_var = 1 / (num_points_in_classes[j] * precs[j][k] + rha)
                means[j][k] = np.random.normal(norm_mean, np.sqrt(norm_var))

        # update precs

        # precompute y_sqsum
        y_sqsum = [[0 for k in range(testing_dimensions)] for j in range(num_classes)]
        for j in range(num_data_points):
            # multi-d
            # x, y = data[j]
            indicator = indicators[j]
            # y_sqsum[indicator][0] += (x - means[indicator][0]) ** 2
            # y_sqsum[indicator][1] += (y - means[indicator][1]) ** 2
            for k in range(testing_dimensions):
                y_sqsum[indicator][k] += (data[j][k] - means[indicator][k]) ** 2

        for j in range(num_classes):
            for k in range(testing_dimensions):
                gam_shape = beta + num_points_in_classes[j]
                gam_scale = ((omega * beta + y_sqsum[j][k]) / (beta + num_points_in_classes[j])) ** -1
                # import pdb; pdb.set_trace()
                precs[j][k] = np.random.gamma(gam_shape, gam_scale)

        # update indicators
        for j in range(num_data_points):
            datapoint = data[j]
            # mixture to sample
            # mixture_prob = np.ones([num_classes + 1])
            mixture_prob = [BigNumber(1)] * (num_classes + 1)

            indicator = indicators[j]
            for k in range(num_classes):
                if indicator == k:
                    nij = num_points_in_classes[k] - 1
                else:
                    nij = num_points_in_classes[k]
                if nij > 0:
                    # case 1
                    mixture_prob[k] *= nij / (num_data_points - 1 + alpha)
                    for z in range(testing_dimensions):
                        try:
                            mixture_prob[k] *= np.sqrt(precs[k][z]) * np.exp(-precs[k][z] / 2 * (datapoint[z] - means[k][z]) ** 2)
                        except TypeError:
                            import pdb; pdb.set_trace()
                            print('hello world')
                else:
                    # case 2
                    mixture_prob[k] *= alpha / (num_data_points - 1 + alpha)
                    for z in range(testing_dimensions):
                        mixture_prob[k] *= np.sqrt(precs[k][z]) * np.exp(-precs[k][z] / 2 * (datapoint[z] - means[k][z]) ** 2)
            # case 3
            mixture_prob[-1] *= alpha / (num_data_points - 1 + alpha)
            # multi-d, not hard, follow case1&2
            # new_mixture_mean_x = np.random.normal(lam, np.sqrt(rha ** -1))
            # new_mixture_prec_x = np.random.gamma(beta, omega ** -1)
            # mixture_prob[-1] *= np.sqrt(new_mixture_prec_x) * np.exp(-new_mixture_prec_x / 2 * (datapoint[0] - new_mixture_mean_x) ** 2)
            # new_mixture_mean_y = np.random.normal(lam, np.sqrt(rha ** -1))
            # new_mixture_prec_y = np.random.gamma(beta, omega ** -1)
            # mixture_prob[-1] *= np.sqrt(new_mixture_prec_y) * np.exp(-new_mixture_prec_y / 2 * (datapoint[1] - new_mixture_mean_y) ** 2)
            # temp arrays for this data point only
            new_mixture_means = [0 for k in range(testing_dimensions)]
            new_mixture_precs = [0 for k in range(testing_dimensions)]
            for k in range(testing_dimensions):
                new_mixture_mean = np.random.normal(lam, np.sqrt(rha ** -1))
                new_mixture_means[k] = new_mixture_mean
                new_mixture_prec = np.random.gamma(beta, omega ** -1)
                new_mixture_precs[k] = new_mixture_prec
                mixture_prob[-1] *= np.sqrt(new_mixture_prec) * np.exp(-new_mixture_prec / 2 * (datapoint[k] - new_mixture_mean) ** 2)
                # import pdb; pdb.set_trace()

            # sample new indicator for this datapoint
            exps = [float(log_base(e, base=math.exp(1))) for e in mixture_prob]
            if debug:
                import pdb; pdb.set_trace()
            sum_exp = logsumexp(exps)
            normal_mixture_prob = [math.exp(e - sum_exp) for e in exps]
            
            # normal_mixture_prob = mixture_prob / np.sum(mixture_prob)
            try:
                cluster_dist = rv_discrete(values=(range(num_classes + 1), normal_mixture_prob))
            except ValueError:
                print('a bad data!!')
                import pdb; pdb.set_trace()
                # return init_indicators, [init_means], [init_precs]
            # import pdb; pdb.set_trace()

            new_indicator = int(cluster_dist.rvs(size=1))
            indicators[j] = new_indicator
            num_points_in_classes[indicator] -= 1

            if new_indicator != num_classes:
                # represented case
                num_points_in_classes[new_indicator] += 1
                # check empty mixtures, if so, update
                # num_classes, num_points_in_classes, indicators, means, precs
            else:
                # unrepresented case
                num_classes += 1
                num_points_in_classes.append(1)
                # multi-d
                # means.append([new_mixture_mean_x, new_mixture_mean_y])
                # precs.append([new_mixture_prec_x, new_mixture_prec_y])
                means.append(new_mixture_means)
                precs.append(new_mixture_precs)
            # print('num_classes ' + str(num_classes))
            # print('num_points_in_classes ' + str(num_points_in_classes))
            # print('indicator ' + str(indicator))
            # print('new_indicator ' + str(new_indicator))
            # print('indicators ' + str(indicators))
            for k in range(num_classes):
                if num_points_in_classes[k] == 0:
                    num_classes -= 1
                    del num_points_in_classes[k]
                    # multi-d, might not need to change
                    del means[k]
                    del precs[k]
                    for z in range(num_data_points):
                        if indicators[z] > k:
                            indicators[z] -= 1
                    # import pdb; pdb.set_trace()
                    break

        # testing
        # print(indicators)

        # end of updating indicators
    # end of sampling

    return str_idx, indicators, means, precs
