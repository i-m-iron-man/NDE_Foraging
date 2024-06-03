import matplotlib.pyplot as plt
import jax.numpy as jnp
from matplotlib import animation, rc
from IPython.display import HTML
from jax.numpy import pi
import seaborn as sns
import diffrax as dfx
from main import BigField
from main import Train

import jax
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
    solver = dfx.EulerHeun()

    fig, ax = plt.subplots()
    ax.set_xlim(( -pi/2, 3*pi/2))#(( -0.1, 2*pi))
    ax.set_ylim(( -0.1, 2*pi))
    ax.set_xlabel('q1')
    ax.set_ylabel('q2')
    tar_pos = jnp.array([[pi/2, pi/2], [pi/2, 3*pi/2]])
    res = ax.scatter(tar_pos[:,0], tar_pos[:,1], c='b', s=100)
    agent = ax.plot(2,3, marker='o', c='r', markersize=10)

    def simData():
        tprev = t0
        tnext = t0 + dt
        args = None
        brownian_term = dfx.VirtualBrownianTree(t0, t1, tol=1e-3, shape = (2,), key = system.random_machine.produce_key())
        term = dfx.MultiTerm(dfx.ODETerm(system.term), dfx.ControlTerm(system.diffusion, brownian_term))

        #dfx.ODETerm(system.term)
        init_state_x_s_e = system.reset(batch_size=1)
        x_init, s_init, e_init = init_state_x_s_e

        print("x_init: ", x_init)
        print("s_init: ", s_init)
        print("e_init: ", e_init)
        obs = jnp.array([jnp.sin(x_init[0][0]), jnp.cos(x_init[0][0]), x_init[0][1],
                     jnp.sin(x_init[0][2]), jnp.cos(x_init[0][2]), x_init[0][3],
                     s_init[0][0], s_init[0][1], e_init[0][0]])
        print("obs: ", obs)
        z_init = system.produce_z(obs)
        #y = (x_init[0], z_init, s_init[0], e_init[0])
        y = jnp.concatenate((x_init[0], z_init, s_init[0], e_init[0]),axis=0)

        state = solver.init(term, tprev, tnext, y, args)
        while tprev < t1:
            y, _, _, state, _ = solver.step(term, tprev, tnext, y, args, state, made_jump=False)
            yield y
            tprev = tnext
            tnext = min(t1, tprev+dt)
            #print("time: ", tprev)
    
    def simPoints(simData):
            x, z, s, e = simData[:4], simData[4:4+system.num_neurons], simData[4+system.num_neurons:4+system.num_neurons+2], simData[4+system.num_neurons+2:4+system.num_neurons+3]
            #simData[0], simData[1], simData[2], simData[3]
            #r_eaten, control = system.term_render((x, z, s, e))
            print("e: ", e)
            agent[0].set_data(jnp.mod(x[0], 2*pi), jnp.mod(x[2],2*pi))
            res.set_sizes(s*20)
            plt.draw()
    anim = animation.FuncAnimation(fig, simPoints, simData, interval=100, repeat=False)
    plt.show(block=False)
    plt.pause(400)
    plt.close()

if __name__ == "__main__":
    system = BigField(seed=11,num_neurons=40)
    train = Train(system, seed=11)
    learned_system = eqx.tree_deserialise_leaves('./models/epoch_870seed_1val9.73478831893567.eqx', system)
    #learned_system = eqx.tree_deserialise_leaves('./models/deterministic.eqx', system)
    render(learned_system)

