import numpy as np
from sklearn.metrics import r2_score
from scipy.optimize import minimize


def neglogli_poissGLM(thetas, xx, yy, dt_bin, vals_to_return=3):
    """ Compute negative log-likelihood of data under Poisson GLM model with
        exponential nonlinearity.

        Args
        ----
        thetas: ndarray (d X 1)
            parameter vector
        xx: ndarray (T X d)
            design matrix
        yy: ndarray (T X 1)
            response variable (spike count per time bin)
        dt_bin: float
            time bin size used
        vals_to_return: int
            which of negative log-likelihood (0), gradient (1), or hessian (2) to return.
            (3) returns all three values. This is necessary due to scipy.optimize.minimize
            requiring the three separate functions with a single return value for each.

        Returns
        -------
        neglogli: float
            negative log likelihood of spike train
        dL: ndarray (d X 1)
            gradient
        H: ndarray (d X d)
            Hessian (second derivative matrix)
    """

    # Compute GLM filter output and conditional intensity
    vv = xx @ thetas # filter output
    # vv = xx @ thetas  # filter output
    rr = np.exp(vv) * dt_bin  # conditional intensity (per bin)

    if len(np.where(np.isnan(rr))[0]) > 0:
        print('at GLM filter output')

    # ---------  Compute log-likelihood -----------
    Trm1 = -vv.T @ yy;  # spike term from Poisson log-likelihood
    Trm0 = np.sum(rr)  # non-spike term
    neglogli = Trm1 + Trm0

    # ---------  Compute Gradient -----------------
    dL1 = -xx.T @ yy  # spiking term (the spike-triggered average)
    dL0 = xx.T @ rr  # non-spiking term
    dL = dL1 + dL0

    if len(np.where(np.isnan(dL0))[0]) > 0:
        print('At gradient')

    # ---------  Compute Hessian -------------------
    H = xx.T @ (xx * np.transpose([rr]))  # non-spiking term

    if len(np.where(np.isnan(H))[0]) > 0:
        print('At hessian')

    if vals_to_return == 3:
        return neglogli, dL, H
    else:
        return [neglogli, dL, H][vals_to_return]


def neglogposterior(thetas, neglogli_fun, Cinv, vals_to_return=3):
    """ Compute negative log-posterior given a negative log-likelihood function
        and zero-mean Gaussian prior with inverse covariance 'Cinv'.

        # Compute negative log-posterior by adding quadratic penalty to log-likelihood

        Args
        ----
        thetas: ndarray (d X 1)
            parameter vector
        neglogli_fun: callable
            function that computes negative log-likelihood, gradient, and hessian.
        Cinv: ndarray (d X d)
            inverse covariance of prior
        vals_to_return: int
            which of negative log-posterior (0), gradient (1), or hessian (2) to return.
            (3) returns all three values. This is necessary due to scipy.optimize.minimize
            requiring the three separate functions with a single return value for each.

        Returns
        -------
        neglogpost: float
            negative log posterior
        grad: ndarray (d X 1)
            gradient
        H: ndarray (d X d)
            Hessian (second derivative matrix)
    """

    neglogpost, grad, H = neglogli_fun(thetas)
    neglogpost = neglogpost + .5 * thetas.T @ Cinv @ thetas
    grad = grad + Cinv @ thetas
    H = H + Cinv

    if vals_to_return == 3:
        return neglogpost, grad, H
    else:
        return [neglogpost, grad, H][vals_to_return]


class Ridge_GLM:

    def __init__(self, l=8, bin_sz=None, w0=None):
        self.l = l
        self.bin_sz = bin_sz
        self.w0 = w0

    def predict(self, X):
        y = np.exp(X @ self.weights) * self.bin_sz
        # y = np.exp(X @ self.weights) * self.bin_sz
        return y

    def predict_spikes(self, X):
        rate = self.predict(X)
        spks = np.random.poisson(np.matrix.transpose(rate))
        return spks

    def fit(self, X, y, **kwargs):
        if self.w0 is None:
            self.w0 = (X.T @ y) / np.sum(y)

        Imat = np.identity(X.shape[1])  # identity matrix of size of filter + const
        Imat[0, 0] = 0

        neglogli_func = lambda prs: neglogli_poissGLM(prs, X, y, self.bin_sz)

        Cinv = self.l * Imat  # set inverse prior covariance
        loss_post_func = lambda prs: neglogposterior(prs, neglogli_func, Cinv, vals_to_return=0)
        grad_post_func = lambda prs: neglogposterior(prs, neglogli_func, Cinv, vals_to_return=1)
        hess_post_func = lambda prs: neglogposterior(prs, neglogli_func, Cinv, vals_to_return=2)
        optimizer = minimize(fun=loss_post_func, x0=self.w0, method='trust-ncg', jac=grad_post_func,
                             hess=hess_post_func,
                             tol=1e-6, options={'disp': False, 'maxiter': 100})
        self.weights = optimizer.x
        print('|', end = '')
        return self

    #     def get_params(self, deep = False):
    #         return {'l':self.l, 'bin_sz':self.bin_sz, 'w0': self.w0}

    def get_params(self, deep=True):
        return {'l': self.l, 'bin_sz': self.bin_sz, 'w0': self.w0}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def score(self, X, y):
        pred = self.predict(X)
        score = r2_score(y, pred)
        return score

