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
        warnings.filterwarnings("ignore", category=RuntimeWarning)
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

# ------------------------------------------------------------------------------

class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        BIC = float('inf')
        best_model = None

        for n_components in range(self.min_n_components, self.max_n_components+1):

            try:

                model = self.base_model(n_components)

                # As base_model can return none
                if model is None:
                    continue

                LogL = model.score(self.X, self.lengths)

                # I am totally uncertain how p should be calculated. I found
                # several references on stackexchange (SE) and some hints in the
                # slack channel.
                #
                # SLACK, SE: p = n_components * (n_components-1)
                # SE: p = (n_components - 1) + 2 * (n_components * self.X.shape[1])
                # WWW: p = (np.power(n_components, 2) - 1) + (2.0 * n_components * len(self.X[0]))
                #
                # My solution is however aligned to a description that I found
                # in section 12.2.4.1 at
                # http://web.science.mq.edu.au/~cassidy/comp449/html/ch12s02.html

                transitions = (n_components - 1) * n_components
                gaussian_means = n_components * len(self.X[0])
                gaussian_covar = gaussian_means

                p = (n_components-1 + transitions + gaussian_means + gaussian_covar)

                BIC_candidate = -2.0 * LogL + p * np.log(len(self.X))

            except:
                BIC_candidate = float('inf')

            if BIC_candidate < BIC:
                BIC = BIC_candidate
                best_model = model

        return best_model


# ------------------------------------------------------------------------------

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        DIC = float('-inf')
        best_model = None

        for n_components in range(self.min_n_components, self.max_n_components+1):

            M = 1.0

            try:
                model = self.base_model(n_components)

                initial_LogL = model.score(self.X, self.lengths)

            except:
                continue

            LogLSum = 0

            for word in self.hwords.keys():
                if word != self.this_word:
                    x2, l2 = self.hwords[word]

                    try:
                        SumLogWord = model.score(x2, l2)
                        M += 1
                    except:
                        SumLogWord = 0

                    LogLSum += SumLogWord

            if M == 1:
                M = float('inf')

            DIC_candidate = initial_LogL - (1/(M-1)) * LogLSum * 1.0

            if DIC_candidate > DIC:
                DIC = DIC_candidate
                best_model = model

        return best_model

# ------------------------------------------------------------------------------

class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # k-fold cross-validation requires at least one train/test split by
        # setting n_splits=2 or more, got n_splits=1.

        if len(self.lengths) < 2:
            return None

        bestAverageLL = float('-inf')
        best_model = None

        split_command = KFold(n_splits=min(len(self.lengths), 3))

        for n_components in range(self.min_n_components, self.max_n_components + 1):

            LLSum = 0
            LLCounter = 1

            for cv_train, cv_test in split_command.split(self.sequences):

                x_train = list()
                x_test = list()

                for element in cv_train:
                    x_train += self.sequences[element]

                for element in cv_test:
                    x_test += self.sequences[element]

                x_train, x_test = np.array(x_train), np.array(x_test)

                length_train, length_test = np.array(self.lengths)[cv_train], np.array(self.lengths)[cv_test]

                try:

                    model = self.base_model(n_components)

                    LogL = model.score(x_test, length_test)

                    LLCounter += 1

                except:

                    LogL = 0

                LLSum += LogL

            LL_candidate = LLSum / (LLCounter+1.0)

            if LL_candidate>bestAverageLL:
                bestAverageLL = LL_candidate
                best_model = model

        return best_model