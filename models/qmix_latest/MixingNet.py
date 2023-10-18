import tensorflow as tf
from tensorflow import keras    
from tensorflow.python.keras.layers import Dense     # for the hidden layer
from tensorflow.python.keras import Sequential
class MixingNet(tf.keras.Model):
    def __init__(self, agent_num, qmix_hidden_dim, obs_dim):
        super(MixingNet, self).__init__()
        self.agent_num = agent_num
        self.obs_dim = obs_dim
        self.qmix_hidden_dim = qmix_hidden_dim
        self.agent_output_dim = 1

        self.hyper_w1 = Dense(units=
            self.qmix_hidden_dim *
            self.agent_num *
            self.agent_output_dim, 
            input_shape=(self.agent_num * self.obs_dim,))
        
        self.hyper_b1 = Dense(units=self.qmix_hidden_dim, 
                              input_shape=(self.agent_num * self.obs_dim,))

        self.hyper_w2 = Dense(units=self.qmix_hidden_dim,
                              input_shape=(self.agent_num * self.obs_dim,))
        
        # inputs = tf.keras.Input(shape=((self.agent_num * self.obs_dim,)))
        
        # self.hyper_b21 = Dense(self.agent_num * self.obs_dim, activation='relu', input_shape=(self.agent_num * self.obs_dim,))
        self.hyper_b21 = Dense(self.qmix_hidden_dim, activation='relu', input_shape=(self.agent_num * self.obs_dim,))
        self.hyper_b2 = Dense(1, input_shape=(self.qmix_hidden_dim,))
        
        # self.hyper_b2 = Sequential([
        #     Dense(self.agent_num * self.obs_dim, activation='relu', input_shape=(self.agent_num * self.obs_dim,)),
        #     Dense(self.qmix_hidden_dim, activation='relu'),
        #     Dense(1)
        # ])

    def call(self, q_values, states, batch_size = 32):  # states的shape为(episode_num, max_episode_len， state_shape)
        # 传入的q_values是三维的，shape为(episode_num, max_episode_len， n_agents)
        q_values = tf.reshape(q_values, (-1, 1, self.agent_num))  # (batch_size, 1, n_agents)
        states = tf.reshape(states, (-1, self.agent_num * self.obs_dim))  # (batch_size, state_shape)
        w1 = tf.abs(self.hyper_w1(states)) 
        b1 = self.hyper_b1(states)  
        
        w1 = tf.reshape(w1, (-1, self.agent_num, self.qmix_hidden_dim))  # (batch_size, n_agents, 64)
        b1 = tf.reshape(b1, (-1, 1, self.qmix_hidden_dim))
        
        hidden = tf.nn.relu(tf.matmul(q_values, w1) + b1)  # (batch_size, 1, 64)
        w2 = tf.abs(self.hyper_w2(states)) 

        b21 = self.hyper_b21(states) 
        # b22 = self.hyper_b22(b21)
        b2 = self.hyper_b2(b21)

        w2 = tf.reshape(w2, (-1, self.qmix_hidden_dim, 1))  # (batch_size, 64, 1)
        b2 = tf.reshape(b2, (-1, 1, 1))  # (batch_size, 1， 1)

        q_total = tf.matmul(hidden, w2) + b2  # (batch_size, 1, 1)
        q_total = tf.reshape(q_total, (batch_size, -1, 1))  # (batch_size, 1, 1)
        return q_total