import tensorflow as tf
from tensorflow.python.keras.layers import Dense 
class MLP(tf.keras.Model):
    def __init__(self, act_num):
        super(MLP, self).__init__()
        self.FC1 = Dense(64, activation=None)
        self.FC2 = Dense(64, activation=None)
        self.FC3 = Dense(act_num, activation=None)

    def call(self, obs):
        result = tf.nn.relu(self.FC1(obs))
        result = tf.nn.relu(self.FC2(result))
        result = self.FC3(result)
        return result
