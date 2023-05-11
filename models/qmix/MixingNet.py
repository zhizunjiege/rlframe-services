import tensorflow as tf


class MixingNet(tf.keras.Model):
    def __init__(self, agent_nets, embed_shape):
        super(MixingNet, self).__init__()
        self.agent_nets = agent_nets
        self.agent_num = len(agent_nets)
        self.embed_shape = embed_shape
        self.timesteps = agent_nets[0].input_shape[1]
        self.agent_output_dim = agent_nets[0].output_shape[-1]
        self.hyper_w1_1 = tf.keras.layers.Dense(
            embed_shape, activation='relu', use_bias=True)
        self.hyper_w1_2 = tf.keras.layers.Dense(
            embed_shape *
            self.agent_num *
            self.agent_output_dim,
            activation='linear',
            use_bias=True)
        self.hyper_b1 = tf.keras.layers.Dense(self.embed_shape)

        self.hyper_w2_1 = tf.keras.layers.Dense(
            self.embed_shape, activation='relu', use_bias=True)
        self.hyper_w2_2 = tf.keras.layers.Dense(
            self.embed_shape, activation='linear', use_bias=True)
        self.hyper_b2 = tf.keras.layers.Dense(1, activation="relu")

    def __call__(self, inputs):
        agents_inputs = inputs[0]
        states = inputs[1]
        masks = inputs[2]
        batch_size = states.shape[0]

        agents_outputs = []
        # for agent_net, agent_input, mask in zip(self.agent_nets, agents_inputs, masks):
        #     agent_out = agent_net(agent_input)
        #     agent_out = tf.multiply(agent_out, mask)
        #     agents_outputs.append(agent_out)
        for agent_index in range(self.agent_num):
            agent_out = self.agent_nets[agent_index](agents_inputs[agent_index])
            agent_out = tf.multiply(agent_out, masks[agent_index])
            agents_outputs.append(agent_out)

        # w1 = tf.abs(self.hyper_w1(states))
        w1 = tf.abs(self.hyper_w1_2(self.hyper_w1_1(states)))

        agents_outputs = tf.concat(agents_outputs, 1)
        agents_outputs = tf.expand_dims(agents_outputs, 1)

        w1 = tf.reshape(w1, [
            batch_size, self.agent_output_dim * self.agent_num, -1])
        b1 = self.hyper_b1(states)
        b1 = tf.reshape(b1, [batch_size, 1, -1])
        hidden = tf.keras.activations.elu(tf.matmul(agents_outputs, w1) + b1)

        # w2 = tf.abs(self.hyper_w2(states))
        w2 = tf.abs(self.hyper_w2_2(self.hyper_w2_1(states)))

        w2 = tf.reshape(w2, [batch_size, self.embed_shape, 1])
        b2 = self.hyper_b2(states)
        b2 = tf.reshape(b2, [batch_size, 1, 1])
        y = tf.matmul(hidden, w2) + b2
        q_tot = tf.reshape(y, [-1, 1])
        return q_tot
