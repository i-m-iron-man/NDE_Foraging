import matplotlib.pyplot as plt
import jax.numpy as jnp
from matplotlib import animation, rc
from IPython.display import HTML
from jax.numpy import pi
import seaborn as sns
import diffrax as dfx
from exp_7_multi_agent.main_g_05 import BigField
from exp_7_multi_agent.main_g_05 import Train


import equinox as eqx

sns.set_theme(style="darkgrid")
palette = "viridis"
sns.set_palette(palette)

from jax import config


config.update("jax_debug_nans", True)
config.update("jax_enable_x64", True)


def render(system: BigField, dt = 0.04, t1 = 40.0):
    t0 = 0.0
    t1 = 40.0
    dt = dt
    solver = dfx.Tsit5()

    fig, ax = plt.subplots()
    ax.set_xlim(( -pi/2, 3*pi/2))#(( -0.1, 2*pi))
    ax.set_ylim(( -0.1, 2*pi))
    ax.set_xlabel('q1')
    ax.set_ylabel('q2')
    tar_pos = jnp.array([[pi/2, pi/2], [pi/2, 3*pi/2]])
    res = ax.scatter(tar_pos[:,0], tar_pos[:,1], c='b', s=100)
    agent1 = ax.plot(2,3, marker='o', c='r', markersize=10)
    agent2 = ax.plot(2,3, marker='o', c='r', markersize=10)

    def simData():
        tprev = t0
        tnext = t0 + dt
        args = None
        term = dfx.ODETerm(system.term)
        (x_init_a1, e_init_a1, x_init_a2, e_init_a2, s_init) = system.reset(batch_size=1)
        obs_a1 = jnp.array([jnp.sin(x_init_a1[0][0]), jnp.cos(x_init_a1[0][0]), x_init_a1[0][1],
                        jnp.sin(x_init_a1[0][2]), jnp.cos(x_init_a1[0][2]), x_init_a1[0][3],
                        s_init[0][0], s_init[0][1], e_init_a1[0][0], 
                        jnp.sin(x_init_a2[0][0]), jnp.cos(x_init_a2[0][0]), x_init_a2[0][1],
                        jnp.sin(x_init_a2[0][2]), jnp.cos(x_init_a2[0][2]), x_init_a2[0][3]])
        
        obs_a2 = jnp.array([jnp.sin(x_init_a2[0][0]), jnp.cos(x_init_a2[0][0]), x_init_a2[0][1],
                        jnp.sin(x_init_a2[0][2]), jnp.cos(x_init_a2[0][2]), x_init_a2[0][3],
                        s_init[0][0], s_init[0][1], e_init_a2[0][0], 
                        jnp.sin(x_init_a1[0][0]), jnp.cos(x_init_a1[0][0]), x_init_a1[0][1],
                        jnp.sin(x_init_a1[0][2]), jnp.cos(x_init_a1[0][2]), x_init_a1[0][3]])
        
        z_init_a1, z_init_a2 = system.produce_z(obs_a1, obs_a2)
        y = (x_init_a1[0], z_init_a1, e_init_a1[0], x_init_a2[0], z_init_a2, e_init_a2[0], s_init[0])
        
        state = solver.init(term, tprev, tnext, y, args)
        while tprev<t1:
            y, _, _, state, _ = solver.step(term, tprev, tnext, y, args, state, made_jump=False)
            yield y
            tprev = tnext
            tnext = min(t1, tprev+dt)
            print("time: ", tprev)
        
    def simPoints(simData):
        x_a1, z_a1, e_a1, x_a2, z_a2, e_a2, s = simData[0], simData[1], simData[2], simData[3], simData[4], simData[5], simData[6]
        r_eaten_a1, r_eaten_a2, control_a1, control_a2 = system.term_render((x_a1, z_a1, e_a1, x_a2, z_a2, e_a2, s))
        agent1[0].set_data(jnp.mod(x_a1[0], 2*pi), jnp.mod(x_a1[2], 2*pi))
        agent2[0].set_data(jnp.mod(x_a2[0], 2*pi), jnp.mod(x_a2[2], 2*pi))
        res.set_sizes(s*20)
        plt.draw()
    anim = animation.FuncAnimation(fig, simPoints, simData, interval=100, repeat=False)
    plt.show(block=False)
    plt.pause(400)
    plt.close()

if __name__ == "__main__":
    system = BigField(seed=3, num_neurons=40)
    train = Train(system, seed=3)
    learned_system = eqx.tree_deserialise_leaves('./models/epoch_976seed_1val187.19970547518386g_07.eqx', system)
    render(learned_system)


