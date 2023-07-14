import numpy as np
from numpy import newaxis as na

from nirdizati_light.predictive_model.common import get_tensor
from nirdizati_light.predictive_model.predictive_model import drop_columns


def lrp_explain(CONF, predictive_model, full_test_df, encoder):
    test_df = drop_columns(full_test_df)
    test_tensor = get_tensor(CONF, test_df)

    explainer = _init_explainer(predictive_model.model, test_tensor[0])
    importances = _get_explanation(explainer, full_test_df, encoder)

    return importances


def _init_explainer(model, tensor):
    return LSTM_bidi(model, tensor)  # load trained LSTM model


def _get_explanation(explainer, target_df, encoder):
    trace_ids = target_df['trace_id'].to_list()
    targets_label_enc = np.argmax(np.array(target_df['label'].to_list()), axis=1)
    df = drop_columns(target_df)
    prefix_names = df.columns
    encoder.decode(df)
    prefixes = df.to_numpy().tolist()

    importances = {}
    for trace_id, prefix_words, label in zip(trace_ids, prefixes, targets_label_enc):
        Rx, Rx_rev, R_rest = explainer.lrp(prefix_words, label)
        words_importances = np.sum(Rx + Rx_rev, axis=1)
        importances[str(trace_id)] = []
        for i_word, word in enumerate(prefix_words):
            importances[str(trace_id)].append([prefix_names[i_word], word, words_importances[i_word]])

    return importances


"""
@author: Leila Arras
@maintainer: Leila Arras
@date: 21.06.2017
@version: 1.0+
@copyright: Copyright (c) 2017, Leila Arras, Gregoire Montavon, Klaus-Robert Mueller, Wojciech Samek
@license: see LICENSE file in repository root
"""


class LSTM_bidi:

    def __init__(self, model, input_encoded):
        """
        Load trained model from file.
        """

        # self.args = CONF

        # input (i.e., one-hot encoded activity + one-hot encoded data attributes)
        self.E = input_encoded

        # model weights
        self.model = model

        """
        Assumptions:
        - bias bxh_left and bxh_right is not stored by Keras
        - bias of output layer is also set to 0
        """

        # LSTM left encoder
        self.Wxh_Left = model.layers[1].get_weights()[0].T  # shape 4d*e // kernel left lstm layer // d = neurons
        # self.bxh_Left = model["bxh_Left"]  # shape 4d; not in Keras
        self.Whh_Left = model.layers[1].get_weights()[1].T  # shape 4d*d // recurrent kernel left lstm layer
        self.bhh_Left = model.layers[1].get_weights()[2].T  # shape 4d // biases left lstm layer

        # LSTM right encoder
        self.Wxh_Right = model.layers[1].get_weights()[3].T  # shape 4d*e // kernel right lstm layer
        # self.bxh_Right = model["bxh_Right"]; not in Keras
        self.Whh_Right = model.layers[1].get_weights()[4].T  # shape 4d*d // recurrent kernel right lstm layer
        self.bhh_Right = model.layers[1].get_weights()[5].T  # shape 4d // biases right lstm layer

        # linear output layer Note: Keras does not provide two output weight vector of the bi-lslm cell;
        # therefore, we divided the vector in two equal parts
        self.Why_Left = model.layers[2].get_weights()[0].T  # shape C*d
        self.Why_Left = self.Why_Left[:, 0:100]
        self.Why_Right = model.layers[2].get_weights()[0].T  # shape C*d
        self.Why_Right = self.Why_Right[:, 100:200]

    def set_input(self, w, delete_pos=None):
        """
        Build the numerical input sequence x/x_rev from the word indices w (+ initialize hidden layers h, c).
        Optionally delete words at positions delete_pos.
        """
        T = len(w)  # prefix length
        d = int(self.Wxh_Left.shape[0] / 4)  # hidden layer dimensions
        e = self.E.shape[1]  # one-hot dimensions; previous, e = self.args.dim
        x = self.E  # encoded input

        if delete_pos is not None:
            x[delete_pos, :] = np.zeros((len(delete_pos), e))

        self.w = w
        self.x = x
        self.x_rev = x[::-1, :].copy()

        self.h_Left = np.zeros((T + 1, d))
        self.c_Left = np.zeros((T + 1, d))
        self.h_Right = np.zeros((T + 1, d))
        self.c_Right = np.zeros((T + 1, d))

    def forward(self):
        """
        Standard forward pass.
        Compute the hidden layer values (assuming input x/x_rev was previously set)
        """
        T = len(self.w)
        d = int(self.Wxh_Left.shape[0] / 4)
        # gate indices (assuming the gate ordering in the LSTM weights is i,g,f,o):
        idx = np.hstack((np.arange(0, d), np.arange(2 * d, 4 * d))).astype(int)  # indices of gates i,f,o together
        idx_i, idx_g, idx_f, idx_o = np.arange(0, d), np.arange(d, 2 * d), np.arange(2 * d, 3 * d), np.arange(3 * d,
                                                                                                              4 * d)  # indices of gates i,g,f,o separately

        # initialize
        self.gates_xh_Left = np.zeros((T, 4 * d))
        self.gates_hh_Left = np.zeros((T, 4 * d))
        self.gates_pre_Left = np.zeros((T, 4 * d))  # gates pre-activation
        self.gates_Left = np.zeros((T, 4 * d))  # gates activation

        self.gates_xh_Right = np.zeros((T, 4 * d))
        self.gates_hh_Right = np.zeros((T, 4 * d))
        self.gates_pre_Right = np.zeros((T, 4 * d))
        self.gates_Right = np.zeros((T, 4 * d))

        for t in range(T):
            self.gates_xh_Left[t] = np.dot(self.Wxh_Left, self.x[t])
            self.gates_hh_Left[t] = np.dot(self.Whh_Left, self.h_Left[t - 1])
            self.gates_pre_Left[t] = self.gates_xh_Left[t] + self.gates_hh_Left[t] + self.bhh_Left  # + self.bxh_Left
            self.gates_Left[t, idx] = 1.0 / (1.0 + np.exp(- self.gates_pre_Left[t, idx]))
            self.gates_Left[t, idx_g] = np.tanh(self.gates_pre_Left[t, idx_g])
            self.c_Left[t] = self.gates_Left[t, idx_f] * self.c_Left[t - 1] + self.gates_Left[t, idx_i] * \
                             self.gates_Left[t, idx_g]
            self.h_Left[t] = self.gates_Left[t, idx_o] * np.tanh(self.c_Left[t])

            self.gates_xh_Right[t] = np.dot(self.Wxh_Right, self.x_rev[t])
            self.gates_hh_Right[t] = np.dot(self.Whh_Right, self.h_Right[t - 1])
            self.gates_pre_Right[t] = self.gates_xh_Right[t] + self.gates_hh_Right[
                t] + self.bhh_Right  # + self.bxh_Right
            self.gates_Right[t, idx] = 1.0 / (1.0 + np.exp(- self.gates_pre_Right[t, idx]))
            self.gates_Right[t, idx_g] = np.tanh(self.gates_pre_Right[t, idx_g])
            self.c_Right[t] = self.gates_Right[t, idx_f] * self.c_Right[t - 1] + self.gates_Right[t, idx_i] * \
                              self.gates_Right[t, idx_g]
            self.h_Right[t] = self.gates_Right[t, idx_o] * np.tanh(self.c_Right[t])

        self.y_Left = np.dot(self.Why_Left, self.h_Left[T - 1])
        self.y_Right = np.dot(self.Why_Right, self.h_Right[T - 1])
        self.s = self.y_Left + self.y_Right

        return self.s.copy()  # prediction scores

    def lrp(self, w, LRP_class, eps=0.001, bias_factor=0.0):
        """
        Perform Layer-wise Relevance Propagation (LRP) forward and especially backward pass.
        Compute the hidden layer relevances by performing LRP for the target class LRP_class
        (according to the papers:
            - https://doi.org/10.1371/journal.pone.0130140
            - https://doi.org/10.18653/v1/W17-5221)
        :param w: words of prefix
        :param LRP_class: target activity
        :param eps: lrp parameter
        :param bias_factor: lrp parameter
        :return: relevance scores
        """

        # forward pass
        self.set_input(w)
        self.forward()

        T = len(self.w)
        d = int(self.Wxh_Left.shape[0] / 4)
        e = self.E.shape[1]  # previously, e = self.args.dim
        C = self.Why_Left.shape[0]  # number of classes
        idx = np.hstack((np.arange(0, d), np.arange(2 * d, 4 * d))).astype(int)  # indices of gates i,f,o together
        idx_i, idx_g, idx_f, idx_o = np.arange(0, d), np.arange(d, 2 * d), np.arange(2 * d, 3 * d), np.arange(3 * d,
                                                                                                              4 * d)  # indices of gates i,g,f,o separately

        # initialize
        Rx = np.zeros(self.x.shape)
        Rx_rev = np.zeros(self.x.shape)

        Rh_Left = np.zeros((T + 1, d))
        Rc_Left = np.zeros((T + 1, d))
        Rg_Left = np.zeros((T, d))  # gate g only
        Rh_Right = np.zeros((T + 1, d))
        Rc_Right = np.zeros((T + 1, d))
        Rg_Right = np.zeros((T, d))  # gate g only

        Rout_mask = np.zeros((C))
        Rout_mask[LRP_class] = 1.0

        # format reminder: lrp_linear(hin, w, b, hout, Rout, bias_nb_units, eps, bias_factor)
        Rh_Left[T - 1] = lrp_linear(self.h_Left[T - 1], self.Why_Left.T, np.zeros((C)), self.s, self.s * Rout_mask,
                                    2 * d, eps, bias_factor, debug=False)
        Rh_Right[T - 1] = lrp_linear(self.h_Right[T - 1], self.Why_Right.T, np.zeros((C)), self.s, self.s * Rout_mask,
                                     2 * d, eps, bias_factor, debug=False)

        for t in reversed(range(T)):
            Rc_Left[t] += Rh_Left[t]
            Rc_Left[t - 1] = lrp_linear(self.gates_Left[t, idx_f] * self.c_Left[t - 1], np.identity(d), np.zeros((d)),
                                        self.c_Left[t], Rc_Left[t], 2 * d, eps, bias_factor, debug=False)
            Rg_Left[t] = lrp_linear(self.gates_Left[t, idx_i] * self.gates_Left[t, idx_g], np.identity(d),
                                    np.zeros((d)), self.c_Left[t], Rc_Left[t], 2 * d, eps, bias_factor, debug=False)
            Rx[t] = lrp_linear(self.x[t], self.Wxh_Left[idx_g].T, self.bhh_Left[idx_g],  # self.bxh_Left[idx_g] +
                               self.gates_pre_Left[t, idx_g], Rg_Left[t], d + e, eps, bias_factor, debug=False)
            Rh_Left[t - 1] = lrp_linear(self.h_Left[t - 1], self.Whh_Left[idx_g].T,
                                        self.bhh_Left[idx_g], self.gates_pre_Left[t, idx_g],  # self.bxh_Left[idx_g] +
                                        Rg_Left[t], d + e, eps, bias_factor, debug=False)

            Rc_Right[t] += Rh_Right[t]
            Rc_Right[t - 1] = lrp_linear(self.gates_Right[t, idx_f] * self.c_Right[t - 1], np.identity(d),
                                         np.zeros((d)), self.c_Right[t], Rc_Right[t], 2 * d, eps, bias_factor,
                                         debug=False)
            Rg_Right[t] = lrp_linear(self.gates_Right[t, idx_i] * self.gates_Right[t, idx_g], np.identity(d),
                                     np.zeros((d)), self.c_Right[t], Rc_Right[t], 2 * d, eps, bias_factor, debug=False)
            Rx_rev[t] = lrp_linear(self.x_rev[t], self.Wxh_Right[idx_g].T,
                                   self.bhh_Right[idx_g], self.gates_pre_Right[t, idx_g],  # self.bxh_Right[idx_g] +
                                   Rg_Right[t], d + e, eps, bias_factor, debug=False)
            Rh_Right[t - 1] = lrp_linear(self.h_Right[t - 1], self.Whh_Right[idx_g].T,
                                         self.bhh_Right[idx_g], self.gates_pre_Right[t, idx_g],
                                         # self.bxh_Right[idx_g] +
                                         Rg_Right[t], d + e, eps, bias_factor, debug=False)

        return Rx, Rx_rev[::-1, :], Rh_Left[-1].sum() + Rc_Left[-1].sum() + Rh_Right[-1].sum() + Rc_Right[-1].sum()


def lrp_linear(hin, w, b, hout, Rout, bias_nb_units, eps, bias_factor=0.0, debug=False):
    """
    LRP for a linear layer with input dim D and output dim M.
    Args:
    - hin:            forward pass input, of shape (D,)
    - w:              connection weights, of shape (D, M)
    - b:              biases, of shape (M,)
    - hout:           forward pass output, of shape (M,) (unequal to np.dot(w.T,hin)+b if more than one incoming layer!)
    - Rout:           relevance at layer output, of shape (M,)
    - bias_nb_units:  total number of connected lower-layer units (onto which the bias/stabilizer contribution is redistributed for sanity check)
    - eps:            stabilizer (small positive number)
    - bias_factor:    set to 1.0 to check global relevance conservation, otherwise use 0.0 to ignore bias/stabilizer redistribution (recommended)
    Returns:
    - Rin:            relevance at layer input, of shape (D,)
    """
    sign_out = np.where(hout[na, :] >= 0, 1., -1.)  # shape (1, M)

    numer = (w * hin[:, na]) + (bias_factor * (b[na, :] * 1. + eps * sign_out * 1.) / bias_nb_units)  # shape (D, M)
    # Note: here we multiply the bias_factor with both the bias b and the stabilizer eps since in fact
    # using the term (b[na,:]*1. + eps*sign_out*1.) / bias_nb_units in the numerator is only useful for sanity check
    # (in the initial paper version we were using (bias_factor*b[na,:]*1. + eps*sign_out*1.) / bias_nb_units instead)

    denom = hout[na, :] + (eps * sign_out * 1.)  # shape (1, M)

    message = (numer / denom) * Rout[na, :]  # shape (D, M)

    Rin = message.sum(axis=1)  # shape (D,)

    if debug:
        print("local diff: ", Rout.sum() - Rin.sum())
    # Note: - local layer relevance conservation if bias_factor==1.0 and bias_nb_units==D (i.e. when only one
    # incoming layer) - global network relevance conservation if bias_factor==1.0 and bias_nb_units set accordingly
    # to the total number of lower-layer connections -> can be used for sanity check

    return Rin
