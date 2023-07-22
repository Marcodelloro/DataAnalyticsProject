
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate
from scipy.integrate import solve_ivp
from sindyfunc import lorenz
import tensorflow as tf
from sklearn.metrics import mean_squared_error

n = 100 #training data size
nt = 20 #testing data size
epochs = 200

# Create training data of Lorenz trajectories: 100 different trajectories to train the NN
dt = 0.01
T = 10
t = np.arange(0,T+dt,dt)
# parameters of Lorenz Dynamics
t_span = (t[0], t[-1])

nn_input = np.zeros((n*(len(t)-1),3)) #Training dataset
nn_output = np.zeros_like(nn_input)     #Testing dataset

nn_in_test = np.zeros((nt*(len(t)-1),3)) #Testing dataset
nn_out_test = np.zeros((nt*(len(t)-1),3))

print(nn_in_test.shape)
print(nn_out_test.shape)

x_t=[] #array initialization
x0 = -15 + 30 * np.random.random((n, 3)) # creates 100 different initial conditions
x_tot = []  #array of all the results coming from the ode integration
print(x0.shape)

for i in range(x0.shape[0]):
    x_t = np.asarray(solve_ivp(lorenz, t_span, x0[i,:], t_eval=t).y)
    x_tot.append(x_t)
    t_ode = solve_ivp(lorenz, t_span, x0[i,:], t_eval=t).t

x_tot = np.asarray(x_tot)
x_tot = np.transpose(x_tot, (0, 2, 1))

# creation of two matrices, one representing the current time instant,
# the other the system advanced of dt (x_k+1)

for j in range(x0.shape[0]):
    nn_input[j*(len(t)-1):(j+1)*(len(t)-1),:] = x_tot[j,:-1,:] # matrix of the system at x_k
    nn_output[j*(len(t)-1):(j+1)*(len(t)-1),:] = x_tot[j,1:,:] # matrix of the system at x_k + 1

# Creation of testing data
for j in range(nt):
    nn_in_test[j*(len(t)-1):(j+1)*(len(t)-1),:] = x_tot[j,:-1,:] # matrix of the system at x_k
    nn_out_test[j*(len(t)-1):(j+1)*(len(t)-1),:] = x_tot[j,1:,:] # matrix of the system at x_k + 1

# # Create figure and 3D axis
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# for i in range(x_tot.shape[0]):
#     ax.scatter(x_tot[i, 0, 0], x_tot[i, 0, 1], x_tot[i, 0, 2], c='r', marker='x')
#     ax.plot3D(x_tot[i, :, 0], x_tot[i, :, 1], x_tot[i, :, 2], linewidth=0.5)
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# plt.title('Training Data')
# plt.show()

# ----------------------------------Training NN --------------------------------------

# The NN must learn the nonlinear mapping from x_k to x_k+1
net = tf.keras.models.Sequential()
net.add(tf.keras.layers.Dense(10, input_dim=3, activation='linear'))
# net.add(tf.keras.layers.Dense(10, activation='relu'))
net.add(tf.keras.layers.Dense(10, activation='relu')) #additional layer
net.add(tf.keras.layers.Dense(3, activation='linear'))
optimizer = tf.keras.optimizers.Adam(lr=0.0001) #lr = 0.001 if standard
net.compile(loss='mse', optimizer=optimizer)
net.summary()
History = net.fit(nn_input, nn_output, validation_data=(nn_in_test,nn_out_test), batch_size=32, epochs=epochs)
loss = History.history['loss']
val_loss = History.history['val_loss']

# test_hist = net.fit(nn_in_test, nn_out_test, epochs=epochs)
# test_loss = test_hist.history['loss']

plt.semilogy(range(1, epochs+1), loss, linewidth=2, label='Train')
plt.semilogy(range(1, epochs+1), val_loss, label='Validation')
plt.grid()
plt.legend()
plt.title('Loss Evaluation')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.show()

print(min(loss))
print(min(val_loss))
# Prediction of future dynamics behaviour
num_traj = 2;
ynn = np.zeros((num_traj, len(t), 3))
ynn[:, 0, :] = -15 + 30 * np.random.random((num_traj, 3))
initial_cond = ynn;
for jj, tval in enumerate(t[:-1]):
    ynn[:, jj + 1, :] = net.predict(ynn[:, jj, :])

x0 =  np.zeros((num_traj, 3))
x0[:,:] = ynn[:,0,:]
t_span = (t[0], t[-1])

x_t=[] #array initialization
x_tot = []  #array of all the results coming from the ode integration

for i in range(x0.shape[0]):
    x_t = np.asarray(solve_ivp(lorenz, t_span, x0[i,:], t_eval=t).y)
    x_tot.append(x_t)
    t_ode = solve_ivp(lorenz, t_span,  x0[i,:], t_eval=t).t

x_tot = np.asarray(x_tot)
x_tot = np.transpose(x_tot, (0, 2, 1))
# initial_cond = ynn;
ax = plt.axes(projection='3d')
ax.plot3D(ynn[0, :, 0], ynn[0, :, 1], ynn[0, :, 2], linewidth=1)
ax.plot3D(ynn[1, :, 0], ynn[1, :, 1], ynn[1, :, 2], linewidth=1)
ax.scatter(ynn[0, 0, 0], ynn[0, 0, 1], ynn[0, 0, 2], label= 'Traj. 1 - Learned')
ax.scatter(ynn[1, 0, 0], ynn[1, 0, 1], ynn[1, 0, 2], label= 'Traj. 1 - Learned')
ax.plot3D(x_tot[0, :, 0], x_tot[0, :, 1], x_tot[0, :, 2], '--', linewidth=0.5, label= 'Traj. 1 - True')
ax.plot3D(x_tot[1, :, 0], x_tot[1, :, 1], x_tot[1, :, 2], '--', linewidth=0.5, label= 'Traj. 2 - True')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.legend(fontsize="7", loc ="best")
plt.title('Reconstructed Lorenz')
plt.show()
mse_net = np.zeros((2,3))
mse_net[0,0]=(mean_squared_error(x_tot[0, :, 0], ynn[0, :, 0],squared=False)) #first row = first trajectory
mse_net[0,1]=(mean_squared_error(x_tot[0, :, 1], ynn[0, :, 1],squared=False))
mse_net[0,2]=(mean_squared_error(x_tot[0, :, 2], ynn[0, :, 2],squared=False))
mse_net[1,0]=(mean_squared_error(x_tot[1, :, 0], ynn[1, :, 0],squared=False))#second row = second trajectory
mse_net[1,1]=(mean_squared_error(x_tot[1, :, 1], ynn[1, :, 1],squared=False))
mse_net[1,2]=(mean_squared_error(x_tot[1, :, 2], ynn[1, :, 2],squared=False))

print(mse_net)

