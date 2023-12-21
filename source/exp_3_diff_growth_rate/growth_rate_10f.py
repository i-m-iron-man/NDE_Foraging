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
        print("self.steepness: ", self.steepness)
        
        self.beta = jnp.array([0.5])
        self.eta = jnp.array([0.1])
        self.gamma = jnp.array([0.01])

        self.x_init = self.random_machine.produce(dim=(4,))
        self.z_init = self.random_machine.produce(dim=(num_neurons,))

        self.s_init = jnp.array([4.0, 4.0])
        self.e_init = jnp.array([0.0])

        self.z_initializer = eqx.nn.Linear(in_features= num_obs, out_features = num_neurons, key=self.random_machine.produce_key())
    
    def term(self, time, state, args):
        (x, z, s, e) = state

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


        return (dx, dz, ds, de)
    
    def term_render(self, state):
        (x, z, s, e) = state

        tanh_z = jnp.tanh(z)
        control = jnp.tanh(jnp.matmul(self.D, tanh_z)) 
        dx = jnp.matmul(self.A, x) + jnp.matmul(self.B, control) 

        O = jnp.array([jnp.sin(x[0]), jnp.cos(x[0]), x[1],
                       jnp.sin(x[2]), jnp.cos(x[2]), x[3],
                       s[0], s[1],e[0]]) 
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


        return (resource_eaten,  control)
    
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
        ys = (sol.ys[0][:-1], sol.ys[1][:-1], sol.ys[2][:-1], sol.ys[3][:-1])
        y1 = jax.lax.stop_gradient(( sol.ys[0][-1] , sol.ys[1][-1] ,  sol.ys[2][-1] , sol.ys[3][-1]))
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
        
        xs = sol[0].reshape(sol[0].shape[0]*sol[0].shape[1],4)
        zs = sol[1].reshape(sol[1].shape[0]*sol[1].shape[1],self.num_neurons)
        ss = sol[2].reshape(sol[2].shape[0]*sol[2].shape[1],2)
        es = sol[3].reshape(sol[3].shape[0]*sol[3].shape[1],1)
        
        final_sol = (xs,zs,ss,es)
        return final_sol, sol[3]


def run_agent(bf:BigField, init_state=None):
    obs = jnp.array([jnp.sin(init_state[0][0]), jnp.cos(init_state[0][0]), init_state[0][1],
                     jnp.sin(init_state[0][2]), jnp.cos(init_state[0][2]), init_state[0][3],
                     init_state[1][0], init_state[1][1], init_state[2][0]])
    z_init = bf.produce_z(obs)
    init_state_x_z_s_e = (init_state[0], z_init, init_state[1], init_state[2])
    sol, xs = bf.simulated_truncated_call(init_state = init_state_x_z_s_e)
    loss = 0.0
    for i in range(len(xs)):
        diff = xs[i][xs[i].shape[0]-1]
        loss += diff
    loss = loss/len(xs)
    return -loss

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
    def __init__(self, bf:BigField, learning_rate = 3e-4, batch_size = 64, t1 = 40.0, dt = 0.04, num_epochs = 2000, best_model_path = "growth_rate_10/", seed = 0):
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
        self.max_value = 10.0

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
                self.max_value = -value+0.1
                self.save(self.best_model_path+"epoch_"+str(epoch)+"seed_"+str(self.seed)+"val"+str(-value)+".eqx")
                print("new best model saved")

        return self.values
    
if __name__ == "__main__":
    seeds = [2]
    values = []
    for seed in seeds:
        test = BigField(num_neurons=40, seed=seed)
        train = Train(test,num_epochs=3000, seed=seed)
        values_per_seed = train.train()
        values.append(values_per_seed)
    values = np.array(values)
   #np.save("values.npy", values)