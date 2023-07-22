import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.cm import rainbow
import numpy as np
from scipy.integrate import solve_ivp
from scipy.io import loadmat
from pysindy.utils import linear_damped_SHO
from pysindy.utils import cubic_damped_SHO
from pysindy.utils import linear_3D
from pysindy.utils import hopf
from pysindy.utils import lorenz
import pysindy as ps
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sindyfunc import paretocurve
from sindyfunc import plot_pareto
from sindyfunc import plot_data_and_derivative
from sindyfunc import plot_data_and_derivative_single
from pysindy.differentiation import FiniteDifference
from pysindy.differentiation import SmoothedFiniteDifference
from pysindy.differentiation import SpectralDerivative
from pysindy.differentiation import SINDyDerivative

print()
# ignore user warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

np.random.seed(1000)  # Seed for reproducibility
# Integrator keywords for solve_ivp
integrator_keywords = {}
integrator_keywords['rtol'] = 1e-12
integrator_keywords['method'] = 'LSODA'
integrator_keywords['atol'] = 1e-12


# Generate training data
dt = 0.002
t_train = np.arange(0, 10, dt) # vector of times for the training data
t_train_span = (t_train[0], t_train[-1]) #from initial to final element of the vector
x0_train = [-8, 8, 27]
x_train = solve_ivp(lorenz, t_train_span,
                    x0_train, t_eval=t_train, **integrator_keywords).y.T
x_dot_train_measured = np.array(
    [lorenz(0, x_train[i]) for i in range(t_train.size)]
)

# Generate testing data
t_test = np.arange(0, 15, dt)
t_test_span = (t_test[0], t_test[-1])
x0_test = np.array([8, 7, 15])
x_test = solve_ivp(
    lorenz, t_test_span, x0_test, t_eval=t_test, **integrator_keywords
).y.T

# Results of the fitting with SINDy

feature_names = ['x', 'y', 'z']

threshold_scan = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
coefs = []
# Add gaussian distribute noise with zero mean
rmse = mean_squared_error(x_train, np.zeros(x_train.shape), squared=False)
x_train_noise = x_train + np.random.normal(0, rmse/50, x_train.shape)
x_train_noise2 = x_train + np.random.normal(0, rmse/50,  x_train.shape) #same data to train the system but w/ noise with larger vari

for i, threshold in enumerate(threshold_scan):
    opt = ps.STLSQ(threshold=threshold)
    model = ps.SINDy(feature_names=feature_names,
                     optimizer=opt)
    model.fit(x_train_noise, t=dt, quiet=True)
    coefs.append(model.coefficients())
res = np.array(coefs)

nonzero = []
threshold_scan2= np.linspace(0,1,10) #goes from 0 to 1
paretocurve(res, threshold_scan2, nonzero)

# evaluation of the mean square error w/ different lambda values

# plot_pareto(coefs, opt, model, threshold_scan, x_test, t_test)




#--------------------- part 2 _ make the algorithm less sensitive to noise ------------------------#
# define the testing and training Lorenz data we will use for these examples

# x_train_noise2 = x_train + np.random.normal(0, rmse/50,  x_train.shape) #same data to train the system but w/ noise with larger variance
# plot_data_and_derivative(x_train_noise, dt,  ps.SmoothedFiniteDifference()._differentiate)
plot_data_and_derivative_single(x_train_noise, dt, ps.FiniteDifference()._differentiate, ps.SmoothedFiniteDifference()._differentiate)

# here the data fitted are causing x_dot to have amplified noise (the differentiation is badly done)
# The way this can be changed is by changing type of DIFFERENTIATION --> ps.FiniteDifference()
#                                                                        ps.SmoothedFiniteDifference()

# Model with standard noise - standard differentiation
mse_findiff = []
mse_smdiff = []
opt1=  ps.STLSQ(threshold=0.1)
dif_method = FiniteDifference(axis=-2)
model1 = ps.SINDy(feature_names=feature_names, optimizer=opt1, differentiation_method=dif_method)
model1.fit(x_train_noise, t=dt)
model1.print()
print('\n')
x_sim = model1.simulate(x0_test, t_test)
mse_findiff.append(mean_squared_error(x_test[:,0], x_sim[:,0]))
mse_findiff.append(mean_squared_error(x_test[:,1], x_sim[:,1]))
mse_findiff.append(mean_squared_error(x_test[:,2], x_sim[:,2]))

dif_method2 = SmoothedFiniteDifference(axis=-2)
model2 = ps.SINDy(feature_names=feature_names, optimizer=opt1, differentiation_method=dif_method2)
model2.fit(x_train_noise, t=dt)
model2.print()
print('\n')
x_sim2 = model2.simulate(x0_test, t_test)
mse_smdiff.append(mean_squared_error(x_test[:,0], x_sim2[:,0]))
mse_smdiff.append(mean_squared_error(x_test[:,1], x_sim2[:,1]))
mse_smdiff.append(mean_squared_error(x_test[:,2], x_sim2[:,2]))

mse_findiff = np.asarray(mse_findiff)
mse_smdiff = np.asarray(mse_smdiff)
print(mse_findiff)
print(mse_smdiff)



#comparison plot of true behaviour vs reconstructed finte diff vs reconstructed smoothdiff

feature_name = ["x", "y", "z"]
for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.plot(x_sim[:, i], label = 'Finite Diff' ) #first simulation
    # plt.plot(x_sim2[:, i], label= 'Smoothed Diff' ) #second simulation
    plt.plot(x_test[:, i], 'r--' , label = 'Ideal Lorenz' ,linewidth=2)  # second simulation
    plt.grid(True)
    plt.xlabel("t", fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
plt.legend(fontsize=10)
plt.suptitle("Dataset Loaded - Lorenz Attractor", fontsize=15)
plt.show()


# Recontruction of the different methods of just the x coord
plt.subplot(1, 2,1)
plt.plot(x_sim[:, 0], label = 'Finite Diff', linewidth=2) #first simulation
plt.plot(x_test[:, 0], 'r--' , label = 'Ideal Lorenz' ,linewidth=1.5)
plt.grid(True)
plt.xlabel("t", fontsize=10)
plt.ylabel("x", fontsize=10)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.legend(fontsize=10)
plt.subplot(1,2,2)
plt.plot(x_sim2[:, 0], 'g', label= 'Smoothed Diff', linewidth=2) #second simulation
plt.plot(x_test[:, 0], 'r--' , label = 'Ideal Lorenz' , linewidth=1.5)
plt.grid(True)
plt.xlabel("t", fontsize=10)
plt.ylabel("x", fontsize=10)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.legend(fontsize=10)
plt.suptitle("Model Comparison Finite VS Smoothed", fontsize=15)
plt.show()


# Try to apply different type of optimizers to see the actual better result

# STSLQ with additional Ridge L2 term "alpha" (all same initial dataset)
opt= ps.STLSQ(threshold= 0.1, alpha=2)
model = ps.SINDy(optimizer=opt)
model.fit(x_train_noise, t=dt)
print('\n')
model.print()

ax = plt.axes(projection='3d')
ax.plot3D(x_train_noise[:, 0], x_train_noise[:, 1], x_train_noise[:, 2], 'blue', label='Input Dataset')
ax.plot3D(x_sim[:, 0], x_sim[:, 1], x_sim[:, 2], 'red', label='Learned Model From Data')
plt.title("Lorenz Attractor - STLS Learned Model")
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_zticklabels([])
ax.set_ylabel('y', fontsize=10)
ax.set_xlabel('x', fontsize=10)
ax.set_zlabel('z', fontsize=10)
ax.legend(fontsize = 7)
plt.show()
mse_STLSQ = []
mse_STLSQ.append(mean_squared_error(x_test[:,0], x_sim[:,0],squared=False))
mse_STLSQ.append(mean_squared_error(x_test[:,1], x_sim[:,1],squared=False))
mse_STLSQ.append(mean_squared_error(x_test[:,2], x_sim[:,2],squared=False))


# SR3 with additional L1 term
opt= ps.SR3(threshold= 0.1, thresholder='l1')
model = ps.SINDy(optimizer=opt)
model.fit(x_train_noise, t=dt)
print('\n')
model.print()
x_sim = model.simulate(x0_test, t_test)
ax = plt.axes(projection='3d')
ax.plot3D(x_train_noise[:, 0], x_train_noise[:, 1], x_train_noise[:, 2], 'blue', label='Input Dataset')
ax.plot3D(x_sim[:, 0], x_sim[:, 1], x_sim[:, 2], 'red', label='Learned Model From Data')
plt.title("Lorenz Attractor - SR3 Learned Model")
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_zticklabels([])
ax.set_ylabel('y', fontsize=10)
ax.set_xlabel('x', fontsize=10)
ax.set_zlabel('z', fontsize=10)
ax.legend(fontsize = 7)
plt.show()
mse_SR3L1 = []
mse_SR3L1.append(mean_squared_error(x_test[:,0], x_sim[:,0],squared=False))
mse_SR3L1.append(mean_squared_error(x_test[:,1], x_sim[:,1],squared=False))
mse_SR3L1.append(mean_squared_error(x_test[:,2], x_sim[:,2],squared=False))

# SR3 with additional Ridge L2 term "alpha" (all same initial dataset)
opt= ps.SR3(threshold= 0.1, thresholder='l2')
model = ps.SINDy(optimizer=opt)
model.fit(x_train_noise, t=dt)
print('\n')
model.print()
x_sim = model.simulate(x0_test, t_test)
ax = plt.axes(projection='3d')
ax.plot3D(x_train_noise[:, 0], x_train_noise[:, 1], x_train_noise[:, 2], 'blue', label='Input Dataset')
ax.plot3D(x_sim[:, 0], x_sim[:, 1], x_sim[:, 2], 'red', label='Learned Model From Data')
plt.title("Lorenz Attractor - SR3 Learned Model")
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_zticklabels([])
ax.set_ylabel('y', fontsize=10)
ax.set_xlabel('x', fontsize=10)
ax.set_zlabel('z', fontsize=10)
ax.legend(fontsize = 7)
plt.show()
mse_SR3L2 = []
mse_SR3L2.append(mean_squared_error(x_test[:,0], x_sim[:,0],squared=False))
mse_SR3L2.append(mean_squared_error(x_test[:,1], x_sim[:,1],squared=False))
mse_SR3L2.append(mean_squared_error(x_test[:,2], x_sim[:,2],squared=False))

# constrained SR3 w/ equality constraints
library = ps.PolynomialLibrary()
library.fit([ps.AxesArray(x_train,{"ax_sample":0,"ax_coord":1})])
n_features = library.n_output_features_
print(f"Features ({n_features}):", library.get_feature_names())
n_targets = x_train.shape[1]
constraint_rhs = np.array([0, 28])

# One row per constraint, one column per coefficient
constraint_lhs = np.zeros((2, n_targets * n_features))

# 1 * (x0 coefficient) + 1 * (x1 coefficient) = 0
constraint_lhs[0, 1] = 1
constraint_lhs[0, 2] = 1

# 1 * (x0 coefficient) = 28
constraint_lhs[1, 1 + n_features] = 1

optimizer = ps.ConstrainedSR3(constraint_rhs=constraint_rhs, constraint_lhs=constraint_lhs)
model = ps.SINDy(optimizer=optimizer, feature_library=library).fit(x_train, t=dt)
model.print()
x_sim = model.simulate(x0_test, t_test)
ax = plt.axes(projection='3d')
ax.plot3D(x_train_noise[:, 0], x_train_noise[:, 1], x_train_noise[:, 2], 'blue', label='Input Dataset')
ax.plot3D(x_sim[:, 0], x_sim[:, 1], x_sim[:, 2], 'red', label='Learned Model From Data')
plt.title("Lorenz Attractor - SR3 Constrained Learned Model")
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_zticklabels([])
ax.set_ylabel('y', fontsize=10)
ax.set_xlabel('x', fontsize=10)
ax.set_zlabel('z', fontsize=10)
ax.legend(fontsize = 7)
plt.show()
mse_SR3c = []
mse_SR3c.append(mean_squared_error(x_test[:,0], x_sim[:,0],squared=False))
mse_SR3c.append(mean_squared_error(x_test[:,1], x_sim[:,1],squared=False))
mse_SR3c.append(mean_squared_error(x_test[:,2], x_sim[:,2],squared=False))

lasso_optimizer = Lasso(alpha=1, max_iter=2000, fit_intercept=False)
model = ps.SINDy(optimizer=lasso_optimizer)
model.fit(x_train, t=dt)
model.print()
x_sim = model.simulate(x0_test, t_test)
ax = plt.axes(projection='3d')
ax.plot3D(x_train_noise[:, 0], x_train_noise[:, 1], x_train_noise[:, 2], 'blue', label='Input Dataset')
ax.plot3D(x_sim[:, 0], x_sim[:, 1], x_sim[:, 2], 'red', label='Learned Model From Data')
plt.title("Lorenz Attractor - Lasso Learned Model")
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_zticklabels([])
ax.set_ylabel('y', fontsize=10)
ax.set_xlabel('x', fontsize=10)
ax.set_zlabel('z', fontsize=10)
ax.legend(fontsize = 7)
plt.show()
mse_lasso = []
mse_lasso.append(mean_squared_error(x_test[:,0], x_sim[:,0],squared=False))
mse_lasso.append(mean_squared_error(x_test[:,1], x_sim[:,1],squared=False))
mse_lasso.append(mean_squared_error(x_test[:,2], x_sim[:,2],squared=False))

# change to array - were lists
mse_SR3L2 = np.asarray(mse_SR3L2)
mse_SR3L1 = np.asarray(mse_SR3L1)
mse_SR3c = np.asarray(mse_SR3c)
mse_STLSQ = np.asarray(mse_STLSQ)
mse_lasso = np.asarray(mse_lasso)

mse_tot = np.vstack((mse_STLSQ, mse_SR3L1, mse_SR3L2, mse_SR3c, mse_lasso))

opti = ['STLSQ $l_2$', 'SR3 $l_1$', 'SR3 $l_2$', 'SR3 Const.', 'Lasso']

fig, ax = plt.subplots()
ax.scatter(opti, mse_tot[:,0], color='blue', label='x', marker ="s")
ax.scatter(opti, mse_tot[:,1], color='red', label='y', marker ="s")
ax.scatter(opti, mse_tot[:,2], color='green', label='z', marker ="s")
# Set labels and title
ax.set_xlabel('optimization methods')
ax.set_ylabel('RMSE')
ax.set_title('RMSE Model vs Simulation')
plt.xticks(rotation='vertical')
ax.xaxis.grid()
ax.yaxis.grid()
ax.set_axisbelow(True)
plt.tight_layout()
plt.legend()
plt.show()