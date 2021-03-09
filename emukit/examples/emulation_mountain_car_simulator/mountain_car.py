import numpy as np
import matplotlib.pyplot as plt
from JSAnimation.IPython_display import display_animation
from matplotlib import animation
from IPython.display import display
from pylab import cm

N_STEPS_MAX = 500


def display_frames_as_gif(frames, title):
    """
    Displays a list of frames as a gif, with controls
    """
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)
    plt.title(title)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=30)
    display(display_animation(anim, default_mode='loop'))


# Define a low fidelity simulation that returns the approximate car dynamics
def simulation(state):
    power = 0.0015
    max_speed = 0.07
    min_position = -1.2
    max_position = 0.6
    position = state[0]
    velocity = state[1]
    action = state[2]
    new_velocity = velocity + power * action - 0.0025 * np.cos(3 * position)
    new_velocity = np.clip(new_velocity, -max_speed, max_speed)
    d_velocity = new_velocity - velocity
    new_position = position + new_velocity
    new_position = np.clip(new_position, min_position, max_position)
    d_position = new_position - position
    return d_position, d_velocity


# Define a low fidelity simulation that returns the approximate car dynamics
def low_cost_simulation(state):
    # A corrupted version of the function above
    power = 0.002
    position = state[0]
    action = state[2]
    d_velocity = power * action - 0.004 * (np.cos(3.3 * position - 0.3))**2 - 0.001
    d_position = d_velocity
    return d_position, d_velocity


def plot_emu_sim_comparison(env, control_params, emulator, fidelity='single'):
    reward, state_trajectory, control_inputs, _ = run_simulation(env, control_params)

    reward_emu, state_trajectory_emu_mean, control_inputs_emu_mean = run_emulation(
        emulator, control_params, state_trajectory[0, :].copy(), fidelity=fidelity)

    f, axarr = plt.subplots(1, 3, figsize=(10, 3))
    h1, = axarr[0].plot(state_trajectory_emu_mean[:, 0])

    h2, = axarr[0].plot(state_trajectory[:, 0])
    axarr[0].set_title('Position')
    axarr[1].plot(state_trajectory_emu_mean[:, 1])
    axarr[1].plot(state_trajectory[:, 1])
    axarr[1].set_title('Velocity')

    axarr[2].plot(control_inputs_emu_mean)
    axarr[2].plot(control_inputs)
    axarr[2].set_title('Control Input')
    f.legend([h1, h2], ['Emulation', 'Simulation'], loc=4)
    plt.tight_layout()
    plt.show()


def run_simulation(env, controller_gains, render=False):
    # Reset environment to starting point
    env.seed(0)
    observation = env.reset()

    # Initialize matrices to store state + control inputs
    state_trajectory = np.ndarray((0, observation.shape[0]))
    control_inputs = np.ndarray((0, env.action_space.shape[0]))
    frames = []
    cost = 0
    for i in range(0, N_STEPS_MAX):
        # Calculate control input
        control_input = calculate_linear_control(observation, controller_gains)
        if render:
            frames.append(env.render(mode='rgb_array'))
        # Save current state + control
        state_trajectory = np.concatenate([state_trajectory, observation[np.newaxis, :]], axis=0)
        control_inputs = np.concatenate([control_inputs, control_input[np.newaxis, :]])

        observation, reward, done, info = env.step(control_input)
        cost -= (reward - 1)
        if done:
            state_trajectory = np.concatenate([state_trajectory, observation[np.newaxis, :]], axis=0)
            return cost, state_trajectory, control_inputs, frames
    return cost, state_trajectory, control_inputs, frames


def run_emulation(dynamics_models, controller_gains, X_0, fidelity='single'):
    observation = X_0.copy()
    state_trajectory = np.ndarray((0, observation.shape[0])) * np.nan
    control_inputs = np.ndarray((0, 1)) * np.nan
    cost = 0

    for _ in range(0, N_STEPS_MAX):
        # Evaluate controller
        control_input = calculate_linear_control(observation, controller_gains)
        cost += (np.power(control_input[0], 2) * 0.1 + 1)

        # Store state + control
        state_trajectory = np.concatenate([state_trajectory, observation[np.newaxis, :]], axis=0)
        control_inputs = np.concatenate([control_inputs, control_input[np.newaxis, :]])

        # Evaluate emulator
        gp_input = np.hstack([observation, control_input])[np.newaxis, :]
        next_state_mean = evaluate_model(dynamics_models[0], gp_input, fidelity)
        observation[0] += next_state_mean
        next_state_mean = evaluate_model(dynamics_models[1], gp_input, fidelity)
        observation[1] += next_state_mean

        if observation[0] > 0.45:
            state_trajectory = np.concatenate([state_trajectory, observation[np.newaxis, :]], axis=0)
            return cost - 100, state_trajectory, control_inputs
    return cost, state_trajectory, control_inputs


def calculate_linear_control(state, gains):
    control_input = (np.dot(gains[0, 0:2], state) + gains[0, 2])[np.newaxis]
    return np.clip(control_input, -1, 1)


def add_data_to_gp(gp_model, new_x, new_y):
    all_X = np.concatenate([gp_model.X, new_x])
    all_y = np.concatenate([gp_model.Y, new_y])
    gp_model.set_XY(all_X, all_y)
    return gp_model


def make_gp_inputs(control_inputs, state_trajectory):
    X = np.concatenate([state_trajectory[:-1, :], control_inputs], axis=1)
    y = np.diff(state_trajectory, axis=0)
    return X, y


def v_simulation(state):
    state = state.copy().flatten()
    power = 0.0015
    max_speed = 0.07
    position = state[0]
    velocity = state[1]
    action = state[2]
    new_velocity = velocity + power * action - 0.0025 * np.cos(3 * position)
    new_velocity = np.clip(new_velocity, -max_speed, max_speed)
    d_velocity = new_velocity - velocity
    return np.asarray([d_velocity])[np.newaxis, :]


class plot_control(object):
    def __init__(self, velocity_emulator, fidelity='single'):
        self.velocity_emulator = velocity_emulator
        self.fidelity = fidelity

    def plot_slices(self, control):
        n_points_contour = 50
        position_contour = np.linspace(-1.2, 0.6, n_points_contour)
        velocity_contour = np.linspace(-1 / 0.07, 1 / 0.07, n_points_contour)
        x_contour_grid = np.meshgrid(position_contour, velocity_contour)
        x_contour = np.ones((n_points_contour**2, 3)) * control
        for i in range(0, len(x_contour_grid)):
            x_contour[:, i] = x_contour_grid[i].flatten()

        # evaluate emulator
        y_emulator = evaluate_model(self.velocity_emulator, x_contour, self.fidelity)

        # evaluate simulator
        y_simulator = np.zeros(x_contour.shape[0])
        for i in range(0, x_contour.shape[0]):
            y_simulator[i] = v_simulation(x_contour[i, :])

        # Do plots
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].set_title('Acceleration from Emulator')
        ax[0].contourf(position_contour, velocity_contour, np.reshape(y_emulator, (n_points_contour, n_points_contour)),
                       cmap=cm.RdBu)
        ax[1].set_title('Acceleration from Simulator')
        ax[1].contourf(position_contour, velocity_contour, np.reshape(y_simulator, (n_points_contour,
                                                                                    n_points_contour)), cmap=cm.RdBu)
        plt.tight_layout()
        ax[1].set_xlabel('Car Position')
        ax[0].set_xlabel('Car Position')
        ax[0].set_ylabel('Car Velocity')


def evaluate_model(model, x, fidelity):
    # Evaluate emulator
    if fidelity == 'single':
        y = model.predict(x)[0]
    elif fidelity == 'multi-linear':
        x_extended = np.hstack([x, np.ones([x.shape[0], 1])])
        y = model.predict(x_extended)[0]
    else:
        raise ValueError('Unknown fidelity')
    return y
