import numpy as np
import scipy  # use numpy if scipy unavailable
import scipy.linalg  # use numpy if scipy unavailable
from utils_registration import run_adjustment_known_matches, fun_known_matches

# The next lines was slightly modify to adapt necesities of DTVIST project.
# All the credits to the author of the original code lines

# Copyright (c) 2004-2007, Andrew D. Straw. All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:

# * Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above
# copyright notice, this list of conditions and the following
# disclaimer in the documentation and/or other materials provided
# with the distribution.

# * Neither the name of the Andrew D. Straw nor the names of its
# contributors may be used to endorse or promote products derived
# from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


def ransac(data, model, n, k, t, d, debug=False, return_all=False):
    """fit model parameters to data using the RANSAC algorithm

This implementation written from pseudocode found at
http://en.wikipedia.org/w/index.php?title=RANSAC&oldid=116358182
{{{
Given:
    data - a set of observed data points
    model - a model that can be fitted to data points
    n - the minimum number of data values required to fit the model
    k - the maximum number of iterations allowed in the algorithm
    t - a threshold value for determining when a data point fits a model
    d - the number of close data values required to assert that a model fits well to data
Return:
    bestfit - model parameters which best fit the data (or nil if no good model is found)
iterations = 0
bestfit = nil
besterr = something really large
while iterations < k {
    maybeinliers = n randomly selected values from data
    maybemodel = model parameters fitted to maybeinliers
    alsoinliers = empty set
    for every point in data not in maybeinliers {
        if point fits maybemodel with an error smaller than t
             add point to alsoinliers
    }
    if the number of elements in alsoinliers is > d {
        % this implies that we may have found a good model
        % now test how good it is
        bettermodel = model parameters fitted to all points in maybeinliers and alsoinliers
        thiserr = a measure of how well model fits these points
        if thiserr < besterr {
            bestfit = bettermodel
            besterr = thiserr
        }
    }
    increment iterations
}
return bestfit
}}}
"""
    iterations = 0
    bestfit = None
    besterr = np.inf
    bestinl = 0
    best_inlier_idxs = None
    while iterations < k:
        maybe_idxs, test_idxs = random_partition(n, data.shape[0])
        maybeinliers = data[maybe_idxs, :]
        test_points = data[test_idxs]
        maybemodel = model.fit(maybeinliers)
        test_err = model.get_error(test_points, maybemodel)
        # select indices of rows with accepted points
        also_idxs = test_idxs[test_err < t]
        alsoinliers = data[also_idxs, :]
        if debug:
            print('test_err.min()', test_err.min())
            print('test_err.max()', test_err.max())
            print('np.mean(test_err)', np.mean(test_err))
            print('iteration %d:len(alsoinliers) = %d' % (
                iterations, len(alsoinliers)))
        if len(alsoinliers) > d:
            betterdata = np.concatenate((maybeinliers, alsoinliers))
            bettermodel = model.fit(betterdata)
            better_errs = model.get_error(betterdata, bettermodel)
            thiserr = np.mean(better_errs)
            if thiserr < besterr:
            #if len(alsoinliers) > bestinl: #this takes best score -> more inliers number
                bestfit = bettermodel
                besterr = thiserr
                bestinl = len(alsoinliers)
                best_inlier_idxs = np.concatenate((maybe_idxs, also_idxs))
        iterations += 1
    if bestfit is None:
        print("did not meet fit acceptance criteria RANSAC")
        #raise ValueError("did not meet fit acceptance criteria")
        #return None
    if return_all:
        if bestfit is None:
            return bestfit, {'inliers': []}
        else:
            return bestfit, {'inliers': best_inlier_idxs}
    else:
        return bestfit


def random_partition(n, n_data):
    """return n random rows of data (and also the other len(data)-n rows)"""
    all_idxs = np.arange(n_data)
    np.random.shuffle(all_idxs)
    idxs1 = all_idxs[:n]
    idxs2 = all_idxs[n:]
    return idxs1, idxs2


class LinearLeastSquaresModel:
    """linear system solved using linear least squares
    This class serves as an example that fulfills the model interface
    needed by the ransac() function.

    """

    def __init__(self, input_columns, output_columns, debug=False):
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.debug = debug

    def fit(self, data):
        A = np.vstack([data[:, i] for i in self.input_columns]).T
        B = np.vstack([data[:, i] for i in self.output_columns]).T
        x, resids, rank, s = scipy.linalg.lstsq(A, B)
        return x

    def get_error(self, data, model):
        A = np.vstack([data[:, i] for i in self.input_columns]).T
        B = np.vstack([data[:, i] for i in self.output_columns]).T
        B_fit = scipy.dot(A, model)
        # sum squared error per row
        err_per_point = np.sum((B-B_fit)**2, axis=1)
        return err_per_point


def is_invertible(A):
    '''
    Check A matrix rank and decides if its is invertible or not
    Parameters
    ----------
    A : numpy.array
        Matrix of dimentions nxn to be checked its invertibility
    Returns
    -------
    is_inv : bool
        Variable which defines if A is invertible or not.
    '''

    is_inv = A.shape[0] == A.shape[1] and np.linalg.matrix_rank(
        A) == A.shape[0]

    return is_inv


class run_adjustment_known_matches_model:
    '''
    This class is defined to register two point clouds using similarity transformation
    and least squares. The class is meant to be used with ransac algorithm.
    '''

    def __init__(self, debug=False):
        self.debug = debug

    def fit(self, data):
        '''
        Find the optimal transformation for the aleatorely given 4 3D-3D correspondences
        Parameters
        ----------
        data : np.array
            Contains the source (x1) and target (x2) point clouds to be registered.
            x1 storaged in the three first columns
            x2 storaged from the fourth column
        Returns
        -------
        T : np.array
            Optimal transformation matrix for the given 4 3D-3D correspondences.
        '''

        # transpose and split data into the two point sets
        x1 = data[:, :3]  # target
        x2 = data[:, 3:]  # source

        # estimate fundamental matrix and return
        x1_ = np.concatenate((x1[:4], np.ones((len(x1[:4]), 1))), axis=1).T
        x2_ = np.concatenate((x2[:4], np.ones((len(x2[:4]), 1))), axis=1).T
        if is_invertible(x2_) == True:
            H = x1_ @ np.linalg.inv(x2_)
        else:
            H = np.eye(4)
        T, _, _, _ = run_adjustment_known_matches(
            x1, x2, H=H, transform='Similarity')

        return T

    def get_error(self, data, T):
        '''
        Computes the error to be optimized. This is the distance between
        the four matched points comming from the two point clouds. Different
        than least squares error -> there is the x and y distance counting by
        two error values
        Parameters
        ----------
        data : np.array
            Contains the source (x1) and target (x2) point clouds to be registered.
            x1 storaged in the three first columns
            x2 storaged from the fourth column
        T : np.array
            Optimal transformation matrix for the given 4 3D-3D correspondences.
        Returns
        -------
        err : float
            Value of the error if T is used for registration.
        '''

        # transpose and split data into the two point
        #data = data.T
        x1 = data[:, :3]  # target
        x2 = data[:, 3:]  # source

        # Same error used in fun
        XXB = np.concatenate((x2, np.ones((len(x2), 1))), axis=1).T
        XXB = np.dot(T, XXB)
        XXB /= XXB[3]

        XB = np.copy(XXB[:3].T)
        XA = np.copy(x1)

        #points_distances_full = np.abs(scipy.spatial.distance.cdist(XA,XB))
        err = np.linalg.norm((XB-XA), axis=1)  # from point cloud to model
        # err = np.abs((XB-XA).reshape(-1)) #from point cloud to model
        #err = np.sum(err)
        # return error per point
        return err
