import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.optimizers import Adam
from buffer import ReplayBuffer
from networks import ActorNetwork, CriticNetwork


class Agent:
    def __init__(self, input_dims, alpha=0.001, beta=0.001, tau=0.005,
                 gamma=0.99, update_actor_interval=2, warmup=500,
                 n_actions=15, max_size=1000000, batch_size=500, noise=0.1):
        self.gamma = gamma
        self.tau = tau
        self.max_action = 1
        self.min_action = -1
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.learn_step_cntr = 0
        self.time_step = 0
        self.warmup = warmup
        self.n_actions = n_actions
        self.update_actor_iter = update_actor_interval

        self.actor = ActorNetwork(n_actions=n_actions, name='actor')

        self.critic_1 = CriticNetwork(name='critic_1')
        self.critic_2 = CriticNetwork(name='critic_2')

        self.target_actor = ActorNetwork(n_actions=n_actions, name='target_actor')
        self.target_critic_1 = CriticNetwork(name='target_critic_1')
        self.target_critic_2 = CriticNetwork(name='target_critic_2')

        self.actor.compile(optimizer=Adam(learning_rate=alpha), loss='mean')
        self.critic_1.compile(optimizer=Adam(learning_rate=beta), loss='mean_squared_error')
        self.critic_2.compile(optimizer=Adam(learning_rate=beta), loss='mean_squared_error')

        self.target_actor.compile(optimizer=Adam(learning_rate=alpha), loss='mean')
        self.target_critic_1.compile(optimizer=Adam(learning_rate=beta), loss='mean_squared_error')
        self.target_critic_2.compile(optimizer=Adam(learning_rate=beta), loss='mean_squared_error')

        self.noise = noise
        self.update_network_parameters(tau=1)

    def choose_action(self, observation, evaluate):
        if evaluate:
            state = tf.convert_to_tensor([observation], dtype=tf.float32)
            mu_prime = self.actor(state)[0]
            mu_prime = tf.clip_by_value(mu_prime, self.min_action, self.max_action)
            self.time_step += 1
            return mu_prime.numpy()

        else:
            if self.time_step < self.warmup:
                mu = np.random.normal(scale=0.75, size=(self.n_actions,))
            else:
                state = tf.convert_to_tensor([observation], dtype=tf.float32)
                mu = self.actor(state)[0]
                #mu = mu.numpy()
                #mu[1::2] += np.random.normal(scale=0.1)
            mu_prime = mu + np.random.normal(scale=self.noise)

            mu_prime = tf.clip_by_value(mu_prime, self.min_action, self.max_action)
            self.time_step += 1

            return mu_prime.numpy()

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        states, actions, rewards, new_states, dones = \
            self.memory.sample_buffer(self.batch_size)

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        states_ = tf.convert_to_tensor(new_states, dtype=tf.float32)

        with tf.GradientTape(persistent=True) as tape:
            target_actions = self.target_actor(states_)
            target_actions = target_actions + \
                tf.clip_by_value(np.random.normal(scale=self.noise), -0.1, 0.1)

            target_actions = tf.clip_by_value(target_actions, self.min_action,
                                              self.max_action)

            q1_ = self.target_critic_1(states_, target_actions)
            q2_ = self.target_critic_2(states_, target_actions)

            q1 = tf.squeeze(self.critic_1(states, actions), 1)
            q2 = tf.squeeze(self.critic_2(states, actions), 1)
            q1_ = tf.squeeze(q1_, 1)
            q2_ = tf.squeeze(q2_, 1)

            critic_value_ = tf.math.minimum(q1_, q2_)
            target = rewards + self.gamma*critic_value_*(1-dones)
            critic_1_loss = keras.losses.MSE(target, q1)
            critic_2_loss = keras.losses.MSE(target, q2)

        critic_1_gradient = tape.gradient(critic_1_loss,
                                          self.critic_1.trainable_variables)
        critic_2_gradient = tape.gradient(critic_2_loss,
                                          self.critic_2.trainable_variables)

        self.critic_1.optimizer.apply_gradients(
                   zip(critic_1_gradient, self.critic_1.trainable_variables))
        self.critic_2.optimizer.apply_gradients(
                   zip(critic_2_gradient, self.critic_2.trainable_variables))

        self.learn_step_cntr += 1

        if self.learn_step_cntr % self.update_actor_iter != 0:
            return

        with tf.GradientTape() as tape:
            new_actions = self.actor(states)
            critic_1_value = self.critic_1(states, new_actions)
            actor_loss = -tf.math.reduce_mean(critic_1_value)

        actor_gradient = tape.gradient(actor_loss,
                                       self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(
                        zip(actor_gradient, self.actor.trainable_variables))

        self.update_network_parameters()


    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        weights = []
        targets = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight * tau + targets[i]*(1-tau))

        self.target_actor.set_weights(weights)

        weights = []
        targets = self.target_critic_1.weights
        for i, weight in enumerate(self.critic_1.weights):
            weights.append(weight * tau + targets[i]*(1-tau))

        self.target_critic_1.set_weights(weights)

        weights = []
        targets = self.target_critic_2.weights
        for i, weight in enumerate(self.critic_2.weights):
            weights.append(weight * tau + targets[i]*(1-tau))

        self.target_critic_2.set_weights(weights)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_weights(self.actor.checkpoint_file)
        self.critic_1.save_weights(self.critic_1.checkpoint_file)
        self.critic_2.save_weights(self.critic_2.checkpoint_file)
        self.target_actor.save_weights(self.target_actor.checkpoint_file)
        self.target_critic_1.save_weights(self.target_critic_1.checkpoint_file)
        self.target_critic_2.save_weights(self.target_critic_2.checkpoint_file)

    def load_models(self):
        print('... loading models ...')
        self.actor.load_weights(self.actor.checkpoint_file)
        self.critic_1.load_weights(self.critic_1.checkpoint_file)
        self.critic_2.load_weights(self.critic_2.checkpoint_file)
        self.target_actor.load_weights(self.target_actor.checkpoint_file)
        self.target_critic_1.load_weights(self.target_critic_1.checkpoint_file)
        self.target_critic_2.load_weights(self.target_critic_2.checkpoint_file)


