import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_bic_score = float("inf")
        best_model = self.base_model(self.n_constant)
        for n in range(self.min_n_components, self.max_n_components + 1):
            try:
                # bic = - 2*log(likelihood) + p*log(N)
                # N is number of data points
                # p => number of parameters
                # p = initial state occupation probabilities
                #     + transition probabilities
                #     + emission probabilities
                # initial state occupation probabilities, can be obtained from n,
                # as the probabilities add up to 1.0 => initial state occupation probabilities = n - 1
                # transition probabilities  => n*(n - 1)
                # emission probabilities => n * num_features * 2 = num_means + num_covar
                # Hence, p = n - 1 + n*(n - 1) + 2* n * num_features
                #         p = n*n + 2*n*num_features - 1

                model = self.base_model(n)
                log_likelihood = model.score(self.X, self.lengths)
                num_features = len(self.X[0])
                log_n = math.log(self.X.shape[0])

                p = n*n + 2*n*num_features - 1
                bic = -2 * log_likelihood + p * log_n
                if bic < best_bic_score:
                    best_bic_score = bic
                    best_model = model
            except:
                continue

        return best_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_dic_score = float("-inf")
        best_model = self.base_model(self.n_constant)

        log_likelihoods = []
        for n in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = self.base_model(n)
                log_likelihood = model.score(self.X, self.lengths)
                log_likelihoods.append(n, log_likelihood)
                log_likelihood_sum = log_likelihood_sum + log_likelihood
            except:
                continue

        m = self.X.shape[0]
        for n, log_likelihood in log_likelihoods:
            try:
                dic_score = log_likelihood - 1 / (m - 1) * (log_likelihood_sum - log_likelihood)
                if dic_score > best_dic_score:
                    best_dic_score = dic_score
                    best_model = self.base_model(n)
            except:
                continue

        return best_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''
    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_cv_score = float("-inf")
        best_model = self.base_model(self.n_constant)
        trained_model = None
        kf = KFold()

        for n in range(self.min_n_components, self.max_n_components + 1):
                if len(self.sequences) < kf.n_splits:
                    break

                log_likelihoods = []

                for train_indexes, test_indexes in kf.split(self.sequences):
                    try:
                        X_train, X_train_length = combine_sequences(train_indexes, self.sequences)
                        X_test, X_test_length = combine_sequences(test_indexes, self.sequences)
                        trained_model = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000,
                                                    random_state=self.random_state,
                                                    verbose=False).fit(X_train, X_train_length)

                        log_likelihood = trained_model.score(X_test, X_test_length)
                        log_likelihoods.append(log_likelihood)
                    except:
                        continue

                cv_score = np.mean(log_likelihoods)
                if cv_score > best_cv_score and trained_model is not None:
                    best_cv_score = cv_score
                    best_model = trained_model

        return best_model
