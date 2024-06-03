import jax
import jax.numpy as jnp
from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5, PIDController
from jax import grad, jit, lax, vmap
from jax.numpy import pi
import equinox as eqx
import optax
import random
from jax import config
config.update("jax_debug_nans", True)
config.update("jax_enable_x64", True)


import numpy as np
from typing import Callable


class RandomMachine():
    key = None
    subkey = None
    initializer = None
    def __init__(self, seed = random.randint(0, 2**32)):
        self.key = jax.random.PRNGKey(seed)
        self.initializer = jax.nn.initializers.glorot_normal()
    def produce(self, dim, scale = 1.0 , min_value= -1.0 , max_value= 1.0, distribution='u'):
        self.key , self.subkey = jax.random.split(self.key)
        if distribution=='u':
            return jnp.multiply(scale,jax.random.uniform(self.subkey, shape=dim, minval=min_value, maxval = max_value))
        elif distribution=='n':
            return jnp.multiply(scale,jax.random.normal(self.subkey, shape=dim))
    def glorot(self, dim):
        self.key , self.subkey = jax.random.split(self.key)
        return self.initializer(self.subkey, dim, jnp.float32)
    def produce_key(self):
        self.key , self.subkey = jax.random.split(self.key)
        return self.subkey
    
class BigField(eqx.Module):
    random_machine: RandomMachine = eqx.static_field()

    z_initializer_a1: Callable
    z_initializer_a2: Callable
    
    num_obs: int = eqx.static_field()
    num_neurons: int = eqx.static_field()
    
    A: jnp.array = eqx.static_field() # dynamics matrix double-integrator
    B: jnp.array = eqx.static_field() # action matrix double-integrator
    
    # learning params for a1
    D_a1: jnp.array # maps mean-firing rate to action, learnable
    tau_inv_a1: jnp.array # inverse time constant, learnable
    E_a1: jnp.array # maps sensors to brain dynamics, learnable
    b_a1: jnp.array # bias, learnable
    J_a1: jnp.array # recurrent connectivity, learnable

    # learning params for a2
    D_a2: jnp.array # maps mean-firing rate to action, learnable
    tau_inv_a2: jnp.array # inverse time constant, learnable
    E_a2: jnp.array # maps sensors to brain dynamics, learnable
    b_a2: jnp.array # bias, learnable
    J_a2: jnp.array # recurrent connectivity, learnable

    # target information
    tar_pos: jnp.array = eqx.static_field() # target position
    sin_tar: jnp.array = eqx.static_field() # sine of target position
    cos_tar: jnp.array = eqx.static_field() # cosine of target position

    steepness: float = eqx.static_field() # steepness of sigmoid

    # environmnt params
    beta: jnp.array = eqx.static_field() # fixed, eating capacity of the agent
    eta: jnp.array = eqx.static_field() # fixed, rate at which resource grows
    gamma : jnp.array = eqx.static_field() # fixed, rate at which resource dies

    #initial state
    x_init_a1:jnp.array=eqx.static_field()
    z_init_a1:jnp.array=eqx.static_field()
    e_init_a1:jnp.array=eqx.static_field()

    x_init_a2:jnp.array=eqx.static_field()
    z_init_a2:jnp.array=eqx.static_field()
    e_init_a2:jnp.array=eqx.static_field()
    s_init:jnp.array=eqx.static_field()

    def __init__(self, num_obs = 15, num_neurons = 70, seed = 0): # extend the obs with position and velocity of the other agent? no
        self.random_machine = RandomMachine(seed)
        self.num_obs = num_obs
        self.num_neurons = num_neurons
        self.A = jnp.array([[0 , 1 , 0, 0], [0, -0.3, 0, 0], [0, 0, 0, 1], [0 ,0, 0, -0.3]])
        self.B = jnp.array([[0, 0], [1, 0], [0, 0], [0, 1]])

        self.D_a1 = self.random_machine.glorot(dim=(2,num_neurons))  #self.random_machine.produce(dim=(2,num_neurons))
        self.tau_inv_a1 = self.random_machine.glorot(dim=(num_neurons,1))
        self.tau_inv_a1 = jnp.reshape(self.tau_inv_a1, (num_neurons,))
        self.E_a1 = self.random_machine.glorot(dim=(num_neurons, num_obs)) #self.random_machine.produce(dim=(num_neurons, num_obs))
        self.b_a1 =  self.random_machine.glorot(dim=(num_neurons,1)) #self.random_machine.produce(dim=(num_neurons,)) #self.random_machine.glorot(dim=(num_neurons,)) #
        self.b_a1 = jnp.reshape(self.b_a1, (num_neurons,))
        self.J_a1 = self.random_machine.glorot(dim=(num_neurons, num_neurons)) #self.random_machine.produce(dim=(num_neurons, num_neurons))

        self.D_a2 = self.random_machine.glorot(dim=(2,num_neurons))  #self.random_machine.produce(dim=(2,num_neurons))
        self.tau_inv_a2 = self.random_machine.glorot(dim=(num_neurons,1))
        self.tau_inv_a2 = jnp.reshape(self.tau_inv_a2, (num_neurons,))
        self.E_a2 = self.random_machine.glorot(dim=(num_neurons, num_obs)) #self.random_machine.produce(dim=(num_neurons, num_obs))
        self.b_a2 =  self.random_machine.glorot(dim=(num_neurons,1)) #self.random_machine.produce(dim=(num_neurons,)) #self.random_machine.glorot(dim=(num_neurons,)) #
        self.b_a2 = jnp.reshape(self.b_a2, (num_neurons,))
        self.J_a2 = self.random_machine.glorot(dim=(num_neurons, num_neurons)) #self.random_machine.produce(dim=(num_neurons, num_neurons))


        self.tar_pos = jnp.array([[pi/2, pi/2], [pi/2, 3*pi/2]])
        
        self.sin_tar = jnp.sin(self.tar_pos)
        self.cos_tar = jnp.cos(self.tar_pos)
        self.steepness = -7.0 # steepnees of gaussian kernel
        print("self.steepness: ", self.steepness)
        
        self.beta = jnp.array([0.5])
        self.eta = jnp.array([0.1, 1.0])
        self.gamma = jnp.array([0.01])

        self.x_init_a1 = self.random_machine.produce(dim=(4,))
        self.z_init_a1 = self.random_machine.produce(dim=(num_neurons,))
        self.e_init_a1 = jnp.array([0.0])

        self.x_init_a2 = self.random_machine.produce(dim=(4,))
        self.z_init_a2 = self.random_machine.produce(dim=(num_neurons,))
        self.e_init_a2 = jnp.array([0.0])

        self.s_init = jnp.array([4.0, 4.0])

        self.z_initializer_a1 = eqx.nn.MLP(in_size= num_obs, out_size = num_neurons, width_size = 64, depth = 1, key=self.random_machine.produce_key())
        self.z_initializer_a2 = eqx.nn.MLP(in_size= num_obs, out_size = num_neurons, width_size = 64, depth = 1, key=self.random_machine.produce_key())    
    
    def term(self, time, state, args):
        (x_a1, z_a1, e_a1, x_a2, z_a2, e_a2, s) = state

        tanh_z_a1 = jnp.tanh(z_a1)
        control_a1 = jnp.tanh(jnp.matmul(self.D_a1, tanh_z_a1)) # (2,) = (2, num_neurons) * (num_neurons,)
        dx_a1 = jnp.matmul(self.A, x_a1) + jnp.matmul(self.B, control_a1) # (4,) = (4,4) * (4,) + (4,2) * (2,)

        tanh_z_a2 = jnp.tanh(z_a2)
        control_a2 = jnp.tanh(jnp.matmul(self.D_a2, tanh_z_a2)) # (2,) = (2, num_neurons) * (num_neurons,)
        dx_a2 = jnp.matmul(self.A, x_a2) + jnp.matmul(self.B, control_a2) # (4,) = (4,4) * (4,) + (4,2) * (2,)

        O_a1 = jnp.array([jnp.sin(x_a1[0]), jnp.cos(x_a1[0]), x_a1[1], jnp.sin(x_a1[2]), jnp.cos(x_a1[2]), x_a1[3], s[0], s[1], e_a1[0],
                         jnp.sin(x_a2[0]), jnp.cos(x_a2[0]), x_a2[1], jnp.sin(x_a2[2]), jnp.cos(x_a2[2]), x_a2[3]]) # (15,)
        O_a2 = jnp.array([jnp.sin(x_a2[0]), jnp.cos(x_a2[0]), x_a2[1], jnp.sin(x_a2[2]), jnp.cos(x_a2[2]), x_a2[3], s[0], s[1], e_a2[0],
                         jnp.sin(x_a1[0]), jnp.cos(x_a1[0]), x_a1[1], jnp.sin(x_a1[2]), jnp.cos(x_a1[2]), x_a1[3]]) # (15,)
        
        EO_a1 = jnp.matmul(self.E_a1, O_a1)
        J_tanh_z_a1 = jnp.matmul(self.J_a1, tanh_z_a1)
        dz_a1 = -z_a1 + EO_a1 + self.b_a1 + J_tanh_z_a1
        dz_a1 = jnp.multiply(dz_a1, (10*jax.nn.sigmoid(0.5*self.tau_inv_a1)))

        EO_a2 = jnp.matmul(self.E_a2, O_a2)
        J_tanh_z_a2 = jnp.matmul(self.J_a2, tanh_z_a2)
        dz_a2 = -z_a2 + EO_a2 + self.b_a2 + J_tanh_z_a2
        dz_a2 = jnp.multiply(dz_a2, (10*jax.nn.sigmoid(0.5*self.tau_inv_a2)))

        sin_agent_a1 = jnp.array([jnp.sin(x_a1[0]), jnp.sin(x_a1[2])])
        cos_agent_a1 = jnp.array([jnp.cos(x_a1[0]), jnp.cos(x_a1[2])])

        sin_agent_a2 = jnp.array([jnp.sin(x_a2[0]), jnp.sin(x_a2[2])])
        cos_agent_a2 = jnp.array([jnp.cos(x_a2[0]), jnp.cos(x_a2[2])])

        sin_dist_a1 = self.sin_tar - sin_agent_a1
        cos_dist_a1 = self.cos_tar - cos_agent_a1
        dual_dist_matrix_a1 = jnp.sqrt(jnp.square(sin_dist_a1) + jnp.square(cos_dist_a1))
        actual_dist_a1 = jnp.sqrt(jnp.sum(jnp.square(dual_dist_matrix_a1), axis=1))

        sin_dist_a2 = self.sin_tar - sin_agent_a2
        cos_dist_a2 = self.cos_tar - cos_agent_a2
        dual_dist_matrix_a2 = jnp.sqrt(jnp.square(sin_dist_a2) + jnp.square(cos_dist_a2))
        actual_dist_a2 = jnp.sqrt(jnp.sum(jnp.square(dual_dist_matrix_a2), axis=1))


        gaussian_a1 = jnp.exp(jnp.multiply(self.steepness, jnp.square(actual_dist_a1)))
        resource_eaten_a1 = jnp.multiply(self.beta, jnp.multiply(gaussian_a1, s))
        gaussian_a2 = jnp.exp(jnp.multiply(self.steepness, jnp.square(actual_dist_a2)))
        resource_eaten_a2 = jnp.multiply(self.beta, jnp.multiply(gaussian_a2, s))

        resource_growth = jnp.multiply(self.eta, s)
        resource_decay = jnp.multiply(self.gamma, jnp.power(s, 2))

        ds = resource_growth - resource_decay - resource_eaten_a1 - resource_eaten_a2
        de_a1 = jnp.array([jnp.sum(resource_eaten_a1)]) - jnp.sum(jnp.square(control_a1))
        de_a2 = jnp.array([jnp.sum(resource_eaten_a2)]) - jnp.sum(jnp.square(control_a2))


        return (dx_a1, dz_a1, de_a1, dx_a2, dz_a2, de_a2, ds)
    
    def term_render(self, state):
        (x_a1, z_a1, e_a1, x_a2, z_a2, e_a2, s) = state

        tanh_z_a1 = jnp.tanh(z_a1)
        control_a1 = jnp.tanh(jnp.matmul(self.D_a1, tanh_z_a1)) # (2,) = (2, num_neurons) * (num_neurons,)
        dx_a1 = jnp.matmul(self.A, x_a1) + jnp.matmul(self.B, control_a1) # (4,) = (4,4) * (4,) + (4,2) * (2,)

        tanh_z_a2 = jnp.tanh(z_a2)
        control_a2 = jnp.tanh(jnp.matmul(self.D_a2, tanh_z_a2)) # (2,) = (2, num_neurons) * (num_neurons,)
        dx_a2 = jnp.matmul(self.A, x_a2) + jnp.matmul(self.B, control_a2) # (4,) = (4,4) * (4,) + (4,2) * (2,)

        O_a1 = jnp.array([jnp.sin(x_a1[0]), jnp.cos(x_a1[0]), x_a1[1], jnp.sin(x_a1[2]), jnp.cos(x_a1[2]), x_a1[3], s[0], s[1], e_a1[0],
                         jnp.sin(x_a2[0]), jnp.cos(x_a2[0]), x_a2[1], jnp.sin(x_a2[2]), jnp.cos(x_a2[2]), x_a2[3]]) # (15,)
        O_a2 = jnp.array([jnp.sin(x_a2[0]), jnp.cos(x_a2[0]), x_a2[1], jnp.sin(x_a2[2]), jnp.cos(x_a2[2]), x_a2[3], s[0], s[1], e_a2[0],
                         jnp.sin(x_a1[0]), jnp.cos(x_a1[0]), x_a1[1], jnp.sin(x_a1[2]), jnp.cos(x_a1[2]), x_a1[3]]) # (15,)
        
        EO_a1 = jnp.matmul(self.E_a1, O_a1)
        J_tanh_z_a1 = jnp.matmul(self.J_a1, tanh_z_a1)
        dz_a1 = -z_a1 + EO_a1 + self.b_a1 + J_tanh_z_a1
        dz_a1 = jnp.multiply(dz_a1, (10*jax.nn.sigmoid(0.5*self.tau_inv_a1)))

        EO_a2 = jnp.matmul(self.E_a2, O_a2)
        J_tanh_z_a2 = jnp.matmul(self.J_a2, tanh_z_a2)
        dz_a2 = -z_a2 + EO_a2 + self.b_a2 + J_tanh_z_a2
        dz_a2 = jnp.multiply(dz_a2, (10*jax.nn.sigmoid(0.5*self.tau_inv_a2)))

        sin_agent_a1 = jnp.array([jnp.sin(x_a1[0]), jnp.sin(x_a1[2])])
        cos_agent_a1 = jnp.array([jnp.cos(x_a1[0]), jnp.cos(x_a1[2])])

        sin_agent_a2 = jnp.array([jnp.sin(x_a2[0]), jnp.sin(x_a2[2])])
        cos_agent_a2 = jnp.array([jnp.cos(x_a2[0]), jnp.cos(x_a2[2])])

        sin_dist_a1 = self.sin_tar - sin_agent_a1
        cos_dist_a1 = self.cos_tar - cos_agent_a1
        dual_dist_matrix_a1 = jnp.sqrt(jnp.square(sin_dist_a1) + jnp.square(cos_dist_a1))
        actual_dist_a1 = jnp.sqrt(jnp.sum(jnp.square(dual_dist_matrix_a1), axis=1))

        sin_dist_a2 = self.sin_tar - sin_agent_a2
        cos_dist_a2 = self.cos_tar - cos_agent_a2
        dual_dist_matrix_a2 = jnp.sqrt(jnp.square(sin_dist_a2) + jnp.square(cos_dist_a2))
        actual_dist_a2 = jnp.sqrt(jnp.sum(jnp.square(dual_dist_matrix_a2), axis=1))


        gaussian_a1 = jnp.exp(jnp.multiply(self.steepness, jnp.square(actual_dist_a1)))
        resource_eaten_a1 = jnp.multiply(self.beta, jnp.multiply(gaussian_a1, s))
        gaussian_a2 = jnp.exp(jnp.multiply(self.steepness, jnp.square(actual_dist_a2)))
        resource_eaten_a2 = jnp.multiply(self.beta, jnp.multiply(gaussian_a2, s))


        return (resource_eaten_a1, resource_eaten_a2, control_a1, control_a2)
    
    def reset(self, batch_size=1):
        x_batch_a1 = jnp.array([self.random_machine.produce(dim=(4,), scale = 3.14)])
        for i in range(batch_size-1):
            x_init = self.random_machine.produce(dim=(1,4), scale = 3.14)
            x_batch_a1 = jnp.append(x_batch_a1, x_init, axis = 0)
        
        x_batch_a2 = jnp.array([self.random_machine.produce(dim=(4,), scale = 3.14)])
        for i in range(batch_size-1):
            x_init = self.random_machine.produce(dim=(1,4), scale = 3.14)
            x_batch_a2 = jnp.append(x_batch_a2, x_init, axis = 0)

        s_init = self.s_init
        s_batch = jnp.tile(s_init, (batch_size, 1))

        e_init_a1 = self.e_init_a1
        e_batch_a1 = jnp.tile(e_init_a1, (batch_size, 1))

        e_init_a2 = self.e_init_a2
        e_batch_a2 = jnp.tile(e_init_a2, (batch_size, 1))

        e_batch = jnp.tile(e_init_a1, (batch_size, 1))

        return (x_batch_a1, e_batch_a1, x_batch_a2, e_batch_a2, s_batch)
    
    def produce_z(self, obs_a1, obs_a2):
        return  self.z_initializer_a1(obs_a1), self.z_initializer_a2(obs_a2)
    

    def simulate_truncated(self, carry, time_info):
        init_state = carry

        term = ODETerm(self.term)
        (interval_begins, interval_endings, ts) = time_info
        solver = Tsit5()
        dt0 = ts[1] - ts[0]
        args=None
        saveat = SaveAt(ts = ts, t1 = True)
        sol = diffeqsolve(term,
                          solver,
                          interval_begins,
                          interval_endings,
                          dt0,
                          init_state,
                          args,
                          saveat=saveat,
                          stepsize_controller= PIDController(rtol=1e-3, atol=1e-6), 
                          max_steps=1000000
                          )
        ys = (sol.ys[0][:-1], sol.ys[1][:-1], sol.ys[2][:-1], sol.ys[3][:-1], sol.ys[4][:-1], sol.ys[5][:-1], sol.ys[6][:-1])
        y1 = jax.lax.stop_gradient(( sol.ys[0][-1] , sol.ys[1][-1] ,  sol.ys[2][-1] , sol.ys[3][-1], sol.ys[4][-1], sol.ys[5][-1], sol.ys[6][-1]))
        return y1,ys
        
    
    def simulated_truncated_call(self, init_state = None, t0 = 0.0, t1 = 40.0, dt = 0.04, density = 1000, K = 200):
        if init_state is None:
            init_state = (self.x_init, self.z_init, self.s_init)
        ts = jnp.linspace(t0,t1,density).reshape(density//K,K)
        dt = ts[0,1] - ts[0,0]
        intervals = jnp.append(0,ts[:,-1] + 0.5*dt)
        carry = init_state
        interval_begins = intervals[:-1]
        interval_endings = intervals[1:]

        carry, sol = jax.lax.scan(self.simulate_truncated, carry, (interval_begins, interval_endings, ts)) # f # carry = (x_init, z_init, s_init) # (t0-0.5(dt) t1+0.5(dt) [t0..........t1])
        
        xs_a1 = sol[0].reshape(sol[0].shape[0]*sol[0].shape[1],4)
        zs_a1 = sol[1].reshape(sol[1].shape[0]*sol[1].shape[1],self.num_neurons)
        es_a1 = sol[2].reshape(sol[2].shape[0]*sol[2].shape[1],1)
        xs_a2 = sol[3].reshape(sol[3].shape[0]*sol[3].shape[1],4)
        zs_a2 = sol[4].reshape(sol[4].shape[0]*sol[4].shape[1],self.num_neurons)
        es_a2 = sol[5].reshape(sol[5].shape[0]*sol[5].shape[1],1)
        ss = sol[6].reshape(sol[6].shape[0]*sol[6].shape[1],2)
        
        final_sol = (xs_a1, zs_a1, es_a1, xs_a2, zs_a2, es_a2, ss)
        return final_sol, sol[2], sol[5]


def run_agent(bf:BigField, init_state=None):
    #init[0] -> x_a1, init[1] -> e_a1, init[2] -> x_a2, init[3] -> e_a2, init[4] -> s
    obs_a1 = jnp.array([jnp.sin(init_state[0][0]), jnp.cos(init_state[0][0]), init_state[0][1],
                        jnp.sin(init_state[0][2]), jnp.cos(init_state[0][2]), init_state[0][3],
                        init_state[4][0], init_state[4][1], init_state[1][0], 
                        jnp.sin(init_state[2][0]), jnp.cos(init_state[2][0]), init_state[2][1],
                        jnp.sin(init_state[2][2]), jnp.cos(init_state[2][2]), init_state[2][3]])
    
    obs_a2 = jnp.array([jnp.sin(init_state[2][0]), jnp.cos(init_state[2][0]), init_state[2][1],
                        jnp.sin(init_state[2][2]), jnp.cos(init_state[2][2]), init_state[2][3],
                        init_state[4][0], init_state[4][1], init_state[3][0], 
                        jnp.sin(init_state[0][0]), jnp.cos(init_state[0][0]), init_state[0][1],
                        jnp.sin(init_state[0][2]), jnp.cos(init_state[0][2]), init_state[0][3]])
    

    z_init_a1, z_init_a2 = bf.produce_z(obs_a1, obs_a2)
    init_state_x_z_e_s = (init_state[0], z_init_a1, init_state[1], init_state[2], z_init_a2, init_state[3], init_state[4])
    sol, es_a1, es_a2 = bf.simulated_truncated_call(init_state = init_state_x_z_e_s)
    loss_a1 = 0.0
    for i in range(len(es_a1)):
        diff = es_a1[i][es_a1[i].shape[0]-1]
        loss_a1 += diff
    loss_a1 = loss_a1/len(es_a1)
    
    loss_a2 = 0.0
    for i in range(len(es_a2)):
        diff = es_a2[i][es_a2[i].shape[0]-1]
        loss_a2 += diff
    loss_a2 = loss_a2/len(es_a2)

    return -loss_a1 - loss_a2

@eqx.filter_value_and_grad
def loss(bf:BigField, init_state_x_s = None):
    losses = jax.vmap(run_agent, in_axes=(None,0))(bf, init_state_x_s)
    return jnp.mean(losses)

@eqx.filter_jit
def make_step(bf:BigField, optimizer, opt_state, init_state_x_s = None):
    value, grads = loss(bf, init_state_x_s)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_bf = eqx.apply_updates(bf, updates)
    return value, new_bf, new_opt_state, optimizer, grads


class Train():
    def __init__(self, bf:BigField, learning_rate = 3e-4, batch_size = 64, t1 = 40.0, dt = 0.04, num_epochs = 2000, best_model_path = "models/", seed = 0):
        self.bf = bf
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.t1 = t1
        self.dt = dt
        self.num_epochs = num_epochs
        self.seed = seed
        self.best_model_path = best_model_path

        
        self.optimizer = optax.chain(optax.clip(1.0), optax.adam(learning_rate))
        self.opt_state = self.optimizer.init(eqx.filter(bf, eqx.is_array_like))
        
        self.values = []
        self.value_at_freq = 0.0
        self.max_value = 100.0

    def save(self, path):
        eqx.tree_serialise_leaves(path, self.bf)
    
    def load(self, path):
        return eqx.tree_deserialise_leaves(path, self.bf)
    
    def train(self, print_feq = 40, render_freq = 2000):
        for epoch in range(self.num_epochs):
            init_state_x_s = self.bf.reset(batch_size=self.batch_size)
            
            value, self.bf, self.opt_state, self.optimizer, grads = make_step(self.bf, self.optimizer, self.opt_state, init_state_x_s)
            self.values.append(-value)
            self.value_at_freq -= value


            if (epoch+1) % print_feq == 0 and epoch != 0:
                print("epoch: ", epoch, "loss: ", self.value_at_freq/print_feq)
                self.value_at_freq = 0.0
            
            if -value > self.max_value:
                self.max_value = -value+20.0
                self.save(self.best_model_path+"epoch_"+str(epoch)+"seed_"+str(self.seed)+"val"+str(-value)+"g_10.eqx")
                print("new best model saved")

        return self.values
    
if __name__ == "__main__":
    seeds = [1,2,3]
    values = []
    for seed in seeds:
        test = BigField(num_neurons=40, seed=seed)
        train = Train(test,num_epochs=1000, seed=seed)
        values_per_seed = train.train()
        values.append(values_per_seed)
    values = np.array(values)
    np.save("values.npy", values)