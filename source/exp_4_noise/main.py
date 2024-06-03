import jax
import jax.numpy as jnp
from diffrax import *
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

    z_initializer: Callable
    
    num_obs: int = eqx.static_field()
    num_neurons: int = eqx.static_field()
    
    A: jnp.array = eqx.static_field() # dynamics matrix double-integrator
    B: jnp.array = eqx.static_field() # action matrix double-integrator
    
    # learning params
    D: jnp.array # maps mean-firing rate to action, learnable
    tau_inv: jnp.array # inverse time constant, learnable
    E: jnp.array # maps sensors to brain dynamics, learnable
    b: jnp.array # bias, learnable
    J: jnp.array # recurrent connectivity, learnable

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
    x_init:jnp.array=eqx.static_field()
    z_init:jnp.array=eqx.static_field()
    s_init:jnp.array=eqx.static_field()
    e_init:jnp.array=eqx.static_field()

    def __init__(self, num_obs = 9, num_neurons = 40, seed = 0):
        self.random_machine = RandomMachine(seed)
        self.num_obs = num_obs
        self.num_neurons = num_neurons

        self.A = jnp.array([[0 , 1 , 0, 0], [0, -0.3, 0, 0], [0, 0, 0, 1], [0 ,0, 0, -0.3]])
        self.B = jnp.array([[0, 0], [1, 0], [0, 0], [0, 1]])

        self.D = self.random_machine.glorot(dim=(2,num_neurons))

        self.tau_inv = self.random_machine.glorot(dim=(num_neurons,1)) 
        self.tau_inv = jnp.reshape(self.tau_inv, (num_neurons,))
        self.E = self.random_machine.glorot(dim=(num_neurons, num_obs)) 
        self.b =  self.random_machine.glorot(dim=(num_neurons,1)) 
        self.b = jnp.reshape(self.b, (num_neurons,))
        self.J = self.random_machine.glorot(dim=(num_neurons, num_neurons))

        self.tar_pos = jnp.array([[pi/2, pi/2], [pi/2, 3*pi/2]])
        
        self.sin_tar = jnp.sin(self.tar_pos)
        self.cos_tar = jnp.cos(self.tar_pos)
        self.steepness = -7.0 # steepnees of gaussian kernel
        
        self.beta = jnp.array([0.5])
        self.eta = jnp.array([0.1])
        self.gamma = jnp.array([0.01])

        self.x_init = self.random_machine.produce(dim=(4,))
        self.z_init = self.random_machine.produce(dim=(num_neurons,))

        self.s_init = jnp.array([4.0, 4.0])
        self.e_init = jnp.array([0.0])

        self.z_initializer = eqx.nn.Linear(in_features= num_obs, out_features = num_neurons, key=self.random_machine.produce_key())
    
    def term(self, time, state, args):
        #(x, z, s, e) = state
        x = state[:4]
        z = state[4:4+self.num_neurons]
        s = state[4+self.num_neurons:4+self.num_neurons+2]
        e = state[4+self.num_neurons+2:]

        tanh_z = jnp.tanh(z)
        control = jnp.tanh(jnp.matmul(self.D, tanh_z)) # (2,) = (2, num_neurons) * (num_neurons,)
        dx = jnp.matmul(self.A, x) + jnp.matmul(self.B, control) # (4,) = (4,4) * (4,) + (4,2) * (2,)

        O = jnp.array([jnp.sin(x[0]), jnp.cos(x[0]), x[1], jnp.sin(x[2]), jnp.cos(x[2]), x[3], s[0], s[1], e[0]]) # (9,)
        
        EO = jnp.matmul(self.E, O)
        J_tanh_z = jnp.matmul(self.J, tanh_z)
        dz = -z + EO + self.b + J_tanh_z
        dz = jnp.multiply(dz, (10*jax.nn.sigmoid(self.tau_inv)))

        sin_agent = jnp.array([jnp.sin(x[0]), jnp.sin(x[2])])
        cos_agent = jnp.array([jnp.cos(x[0]), jnp.cos(x[2])])

        sin_dist = self.sin_tar - sin_agent
        cos_dist = self.cos_tar - cos_agent
        dual_dist_matrix = jnp.sqrt(jnp.square(sin_dist) + jnp.square(cos_dist))
        actual_dist = jnp.sqrt(jnp.sum(jnp.square(dual_dist_matrix), axis=1))

        # gaussian kernel
        gaussian = jnp.exp(jnp.multiply(self.steepness, jnp.square(actual_dist)))
        resource_eaten = jnp.multiply(self.beta, jnp.multiply(gaussian, s))

        resource_growth = jnp.multiply(self.eta, s)
        resource_decay = jnp.multiply(self.gamma, jnp.power(s, 2))

        ds = resource_growth - resource_decay - resource_eaten
        de = jnp.array([jnp.sum(resource_eaten)]) - jnp.sum(jnp.square(control))
        dstate = jnp.concatenate((dx, dz, ds, de), axis = 0)


        return dstate
    
    def diffusion(self, time, state, args):
        x_noise_control = jnp.zeros((4,2))#0.1*jnp.eye(4)
        z_noise_control = jnp.zeros((self.num_neurons,2))
        s_noise_control = 0.1*jnp.eye(2)#jnp.zeros((2,4))
        e_noise_control = jnp.zeros((1,2))#jnp.zeros((1,4))
        return jnp.concatenate((x_noise_control, z_noise_control, s_noise_control, e_noise_control), axis=0)
    
    def reset(self, batch_size=1):
        x_batch = jnp.array([self.random_machine.produce(dim=(4,), scale = 3.14)])# + jnp.array([pi/2,0.0])#self.x_init
        for i in range(batch_size-1):
            x_init = self.random_machine.produce(dim=(1,4), scale = 3.14)# + jnp.array([pi/2,0.0]) #self.x_init
            x_batch = jnp.append(x_batch, x_init, axis = 0)
        
        s_init = self.s_init
        s_batch = jnp.tile(s_init, (batch_size, 1))

        e_init = self.e_init
        e_batch = jnp.tile(e_init, (batch_size, 1))
        return (x_batch, s_batch, e_batch)
    
    def produce_z(self, obs):
        return  self.z_initializer(obs)
    

    def simulate_truncated(self, carry, time_info):
        init_state, key = carry
        x, z, s, e = init_state
        key, subkey = jax.random.split(key)
        
        init_state_array = jnp.concatenate((x,z,s,e), axis = 0)
        term = ODETerm(self.term)
        
        (interval_begins, interval_endings, ts) = time_info
        
        brownian = VirtualBrownianTree(interval_begins, interval_endings, tol = 1e-3, shape = (2,), key=subkey)
        system = MultiTerm(term, ControlTerm(self.diffusion, brownian))
        solver = EulerHeun()#Tsit5()
        dt0 = ts[1] - ts[0]
        args=None
        saveat = SaveAt(ts = ts, t1 = True)
        sol = diffeqsolve(system,
                          solver,
                          interval_begins,
                          interval_endings,
                          dt0,
                          init_state_array,
                          args,
                          saveat=saveat,
                          max_steps=1000000
                          )
        xs, zs, ss, es = (sol.ys[:,:4], sol.ys[:,4:4+self.num_neurons], sol.ys[:,4+self.num_neurons:4+self.num_neurons+2], sol.ys[:,4+self.num_neurons+2:] )
        ys = (xs[:-1], zs[:-1], ss[:-1], es[:-1])
        y1 = jax.lax.stop_gradient((xs[-1], zs[-1], ss[-1], es[-1]))
        key = jax.lax.stop_gradient(key)
        return (y1,key),ys
        
    
    def simulated_truncated_call(self, init_state = None, t0 = 0.0, t1 = 40.0, dt = 0.04, density = 1000, K = 200, key = jax.random.PRNGKey(0)):
        if init_state is None:
            init_state = (self.x_init, self.z_init, self.s_init)
        ts = jnp.linspace(t0,t1,density).reshape(density//K,K)
        dt = ts[0,1] - ts[0,0]
        intervals = jnp.append(0,ts[:,-1] + 0.5*dt)
        carry = (init_state, key)
        interval_begins = intervals[:-1]
        interval_endings = intervals[1:]

        carry, sol = jax.lax.scan(self.simulate_truncated, carry, (interval_begins, interval_endings, ts)) # f # carry = (x_init, z_init, s_init) # (t0-0.5(dt) t1+0.5(dt) [t0..........t1])
        
        xs = sol[0].reshape(sol[0].shape[0]*sol[0].shape[1],4)
        zs = sol[1].reshape(sol[1].shape[0]*sol[1].shape[1],self.num_neurons)
        ss = sol[2].reshape(sol[2].shape[0]*sol[2].shape[1],2)
        es = sol[3].reshape(sol[3].shape[0]*sol[3].shape[1],1)
        
        final_sol = (xs,zs,ss,es)
        return final_sol, sol[3]


def run_agent(bf:BigField, key, init_state=None):
    obs = jnp.array([jnp.sin(init_state[0][0]), jnp.cos(init_state[0][0]), init_state[0][1],
                     jnp.sin(init_state[0][2]), jnp.cos(init_state[0][2]), init_state[0][3],
                     init_state[1][0], init_state[1][1], init_state[2][0]])
    z_init = bf.produce_z(obs)
    init_state_x_z_s_e = (init_state[0], z_init, init_state[1], init_state[2])
    sol, xs = bf.simulated_truncated_call(init_state = init_state_x_z_s_e, key=key)
    loss = 0.0
    for i in range(len(xs)):
        diff = xs[i][xs[i].shape[0]-1]
        loss += diff
    loss = loss/len(xs)
    return -loss

@eqx.filter_value_and_grad
def loss(bf:BigField, keys, init_state_x_s = None):
    losses = jax.vmap(run_agent, in_axes=(None,0,0))(bf, keys, init_state_x_s)
    return jnp.mean(losses)

@eqx.filter_jit
def make_step(bf:BigField, optimizer, opt_state, keys, init_state_x_s = None):
    value, grads = loss(bf, keys, init_state_x_s)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_bf = eqx.apply_updates(bf, updates)
    return value, new_bf, new_opt_state, optimizer, grads


def infer(bf:BigField, keys, init_state=None):
    def per_initial_state(bf, init_state, key):
        solver = EulerHeun()
        obs = jnp.array([jnp.sin(init_state[0][0]), jnp.cos(init_state[0][0]), init_state[0][1],
                     jnp.sin(init_state[0][2]), jnp.cos(init_state[0][2]), init_state[0][3],
                     init_state[1][0], init_state[1][1], init_state[2][0]])
        z_init = bf.produce_z(obs)
        init_state_array = jnp.concatenate([init_state[0], z_init, init_state[1], init_state[2]], axis=0)
        t0 = 0.0
        t1 = 40.0
        dt0 = 0.04
        ts = jnp.linspace(t0,t1,1000)
        args = None
        saveat = SaveAt(ts = ts, t1 = True)
        brownian_term = VirtualBrownianTree(t0, t1, tol=1e-3, shape = (2,), key = key)
        system = MultiTerm(ODETerm(bf.term), ControlTerm(bf.diffusion, brownian_term))
        sol = diffeqsolve(system,
                            solver,
                            t0,
                            t1,
                            dt0,
                            init_state_array,
                            args,
                            saveat=saveat, 
                            max_steps=1000000
                            )
        return sol.ys
    jit_per_initial_state = jax.jit(per_initial_state)
    sol_ys = jax.vmap(jit_per_initial_state, in_axes=(None,0,0))(bf, init_state, keys)
    xs, zs ,ss, es = (sol_ys[:,:,:4], sol_ys[:,:,4:4+bf.num_neurons], sol_ys[:,:,4+bf.num_neurons:4+bf.num_neurons+2], sol_ys[:,:,4+bf.num_neurons+2:4+bf.num_neurons+3])
     # MSE error for the distance to the target when the agent is at the target
    error=[]
    tar_pos = bf.tar_pos # tar_pos = jnp.array([[pi/2, pi/2], [pi/2, 3*pi/2]])
    for xs_traj in xs:
        running_error_dist_1 = []
        running_error_dist_2 = []
        counter_dist_1 = 0
        counter_dist_2 = 0
        for i in range(len(xs_traj)):
            agent_x = xs_traj[i][0]
            agent_y = xs_traj[i][2]
            dist_tar_1 = jnp.sqrt(jnp.square(agent_x - tar_pos[0][0]) + jnp.square(agent_y - tar_pos[0][1]))
            dist_tar_2 = jnp.sqrt(jnp.square(agent_x - tar_pos[1][0]) + jnp.square(agent_y - tar_pos[1][1]))
            
            if dist_tar_1 < 0.5:
                counter_dist_1+=1
                running_error_dist_1.append(dist_tar_1)
            else:
                counter_dist_1 = 0
                running_error_dist_1 = []
            if counter_dist_1 >= 10:
                error.append(jnp.mean(jnp.array(running_error_dist_1)))
            
            if dist_tar_2 < 0.5:
                counter_dist_2+=1
                running_error_dist_2.append(dist_tar_2)
            else:
                counter_dist_2 = 0
                running_error_dist_2 = []
            if counter_dist_2 >= 10:
                error.append(jnp.mean(jnp.array(running_error_dist_2)))
    if len(error) == 0:
        print("No position control")
        error = jnp.array([0.5])
    else:
        error = jnp.array(error)
        print("MSE error for the distance to the target when the agent is at the target: ", jnp.mean(error))
    return jnp.mean(error)


class Train():
    def __init__(self, bf:BigField, learning_rate = 3e-4, batch_size = 64, t1 = 40.0, dt = 0.04, num_epochs = 2000, best_model_path = "models_debug/", seed = 0):
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
        self.max_value = 4.0
        infer_key = self.bf.random_machine.produce_key()
        key, *infer_keys = jax.random.split(infer_key, 6)
        self.infer_keys = jnp.array(infer_keys)
        self.infer_x_s_e = self.bf.reset(batch_size=5)

    def save(self, path):
        eqx.tree_serialise_leaves(path, self.bf)
    
    def load(self, path):
        return eqx.tree_deserialise_leaves(path, self.bf)
    
    def train(self, print_feq = 10, render_freq = 2000):
        key = self.bf.random_machine.produce_key()
        infer_errors = []
        for epoch in range(self.num_epochs):
            init_state_x_s = self.bf.reset(batch_size=self.batch_size)
            key, *keys = jax.random.split(key, self.batch_size+1)
            keys = jnp.array(keys)
            value, self.bf, self.opt_state, self.optimizer, grads = make_step(self.bf, self.optimizer, self.opt_state, keys, init_state_x_s)
            self.values.append(-value)
            self.value_at_freq -= value


            if (epoch+1) % print_feq == 0 and epoch != 0:
                print("epoch: ", epoch, "loss: ", self.value_at_freq/print_feq)
                self.value_at_freq = 0.0
                infer_error = infer(self.bf, self.infer_keys, self.infer_x_s_e)
                infer_errors.append(infer_error)
            
            if -value > self.max_value:
                self.max_value = -value+0.5
                self.save(self.best_model_path+"epoch_"+str(epoch)+"seed_"+str(self.seed)+"val"+str(-value)+".eqx")
                print("new best model saved")

        return self.values, jnp.array(infer_errors)
    
if __name__ == "__main__":
    seeds = [1,2,3]
    values = []
    infer_errors = []
    for seed in seeds:
        test = BigField(num_neurons=40, seed=seed)
        train = Train(test,num_epochs=1000, seed=seed)
        values_per_seed, infer_errors_per_seed = train.train()
        values.append(values_per_seed)
        infer_errors.append(infer_errors_per_seed)
    values = np.array(values)
    infer_errors = np.array(infer_errors)
    print(values.shape)
    print(infer_errors.shape)
    np.save("values.npy", values)
    np.save("infer_errors.npy", infer_errors)