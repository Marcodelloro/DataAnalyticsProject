# SINDY SCRIPT FUNCTIONS - Already written or written by me, gathered in this file
# The file works with the main script sindy-main

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.integrate import solve_ivp
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import Lars
from sklearn.linear_model import LassoLars
from pysindy.utils import lorenz, lorenz_control, enzyme
import pysindy as ps
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d
from matplotlib.ticker import FuncFormatter
from scipy.integrate import odeint
from scipy.special import legendre, chebyt



def paretocurve (res,threshold_scan,nonzero):
    for i in range(res.shape[2]):
        nonzero.append(np.count_nonzero(res[i]))
    # print of the PARETO CURVE showing the behaviour of the different coefficients to different lambdas
    plt.plot(threshold_scan, nonzero, linewidth=3)
    plt.ylabel(r"# coefficients", fontsize=12)
    plt.xlabel(r"$\lambda$", fontsize=12)
    plt.title(r"Pareto Curve of $\lambda$ Coefficients", fontsize=15)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True)
    plt.show()



# Make coefficient plot for threshold scan
def plot_pareto(coefs, opt, model, threshold_scan, x_test, t_test):
    dt = t_test[1] - t_test[0]
    mse = np.zeros(len(threshold_scan))
    mse_sim = np.zeros(len(threshold_scan))
    for i in range(len(threshold_scan)):
        opt.coef_ = coefs[i]
        mse[i] = model.score(x_test, t=dt, metric=mean_squared_error)
        x_test_sim = model.simulate(x_test[0, :], t_test, integrator="odeint")
        if np.any(x_test_sim > 1e4):
            x_test_sim = 1e4
        mse_sim[i] = np.sum((x_test - x_test_sim) ** 2)
    plt.figure()
    plt.semilogy(threshold_scan, mse, "bo")
    plt.semilogy(threshold_scan, mse, "b")
    plt.ylabel(r"$\dot{X}$ RMSE", fontsize=15)
    plt.xlabel(r"$\lambda$", fontsize=15)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title("RMS Error - Dynamics Reconstruction", fontsize=15)
    plt.grid(True)
    plt.show()
    plt.figure()
    plt.semilogy(threshold_scan, mse_sim, "bo")
    plt.semilogy(threshold_scan, mse_sim, "b")
    plt.ylabel(r"$\dot{X}$ RMSE", fontsize=20)
    plt.xlabel(r"$\lambda$", fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True)


def plot_data_and_derivative(x, dt, deriv):
    feature_name = ["$x$", "$y$", "$z$"]
    feature_name2 = ["$\dot{x}$", "$\dot{y}$", "$\dot{z}$"]
    x_dot = deriv(x, t=dt)

    # choose what you need, if want to plot 3 figures x,y,z use #1 otherwise use #2
    fig, axs = plt.subplots(2, 3) #1

    for i in range(3):
        axs[0, i].plot(x[:, i], label=feature_name[i])
        axs[1, i].plot(x_dot[:, i],'red', label=feature_name2[i])
        axs[0, i].legend()
        axs[1, i].legend()
        axs[0, i].set_ylabel(feature_name[i])
        axs[0, i].grid(True)
        axs[1, i].set_xlabel('t')
        axs[1, i].set_ylabel(feature_name2[i])
        axs[1, i].grid(True)
        axs[1, i].yaxis.set_label_coords(-0.1, 0.5)
        axs[0, i].yaxis.set_label_coords(-0.1, 0.5)

        axs[1, i].xaxis.set_tick_params(pad=0.3)
        axs[1, i].yaxis.set_tick_params(pad=0.3)
        axs[0, i].xaxis.set_tick_params(pad=0.3)
        axs[0, i].yaxis.set_tick_params(pad=0.3)
        axs[1, i].xaxis.set_tick_params(labelsize=6)
        axs[1, i].yaxis.set_tick_params(labelsize=6)
        axs[0, i].xaxis.set_tick_params(labelsize=6)
        axs[0, i].yaxis.set_tick_params(labelsize=6)
        plt.gca().get_xaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x / 1000), ',')))
    plt.suptitle("Lorenz Attractor - Smoothed Finite Difference", fontsize=15)
    fig.align_xlabels()
    fig.align_ylabels()
    plt.show()

def plot_data_and_derivative_single(x, dt, deriv, deriv2):
    feature_name = ["$x$"]
    feature_name2 = ["$\dot{x}$"]
    x_dot = deriv(x, t=dt)
    x_dot2 = deriv2(x, t=dt)

    # choose what you need, if want to plot 3 figures x,y,z use #1 otherwise use #2
    plt.subplot(1, 3, 1)
    plt.plot(x[:, 0], label='Measured x', linewidth=2)  # first simulation
    plt.grid(True)
    plt.xlabel("t", fontsize=10)
    plt.ylabel("x", fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(fontsize=10)
    plt.title('Measured Dataset')

    plt.subplot(1, 3, 2)
    plt.plot(x_dot[:, 0], 'r', label='$\dot{x}$', linewidth=2)  # first simulation
    plt.grid(True)
    plt.xlabel("t", fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(fontsize=10)
    plt.title('Finite Differentiation')

    plt.subplot(1, 3, 3)
    plt.plot(x_dot2[:, 0], 'g', label='$\dot{x}$', linewidth=2)  # first simulation
    plt.grid(True)
    plt.xlabel("t", fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(fontsize=10)
    plt.title('Smoothed  Finite Differentiation')
    plt.suptitle("Differentiation Methods", fontsize=15)
    plt.show()



    # x_dot = deriv(x, t=dt)
    # for i in range(3):
    #     plt.subplot(2, 3, i + 1)
    #
    #     plt.plot(x_dot[:, i], label=r"$\dot{" + feature_name[i] + "}$")
    #     plt.grid(True)
    #     plt.xlabel("t", fontsize=10)
    #     plt.xticks(fontsize=10)
    #     plt.yticks(fontsize=10)
    #     plt.legend(fontsize=15)
    #     plt.gca().get_xaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x / 1000), ',')))
    # plt.suptitle("Derivates - SmoothedFiniteDifference Results", fontsize=15)
    # plt.show()


# ----------------------------------------- NN Functions -----------------------------------------------------

def lorenz(t, x, sigma=10, beta=8/3, rho=28):
    return [
        sigma * (x[1] - x[0]),
        x[0] * (rho - x[2]) - x[1],
        x[0] * x[1] - beta * x[2],
    ]

