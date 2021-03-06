"""
Tensorflow test.

@author huayunfly@126.com
@date 2018.06.20
"""


import unittest
import datetime
import numpy as np
import tensorflow as tf

class TensorflowTest(unittest.TestCase):
    """
    Basic test cases.
    """

    def test_a_graph(self):
        """
        Creates a graph and runs it in a session.
        """
        x = tf.Variable(3, name='x')
        y = tf.Variable(4, name='y')
        f = x*x*y + y + 2
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            init.run()
            result = f.eval() # means tf.get_default_session().run(f)
        self.assertEqual(result, 42)

        with tf.Session() as sess1:
            sess1.run(x.initializer)
            sess1.run(y.initializer)
            result1 = sess1.run(f)
        self.assertEqual(result1, 42)

    def test_mgr_graph(self):
        """
        Manages the default and defined graphs.
        """
        x1 = tf.Variable(1)
        graph = tf.Graph()
        with graph.as_default():
            x2 = tf.Variable(2)
        self.assertTrue(x1.graph is tf.get_default_graph())
        self.assertTrue(x2.graph is graph)

    def test_node_life(self):
        """
        Creates a simple graph. Nodes x, w in one session will be
        calculated twice (not reuseble).
        """
        w = tf.constant(3)
        x = w + 2
        y = x + 3
        z = y + 5
        with tf.Session() as sess:
            y.eval()
            z.eval()
        # Tell tf to evaluate both y and z in just one graph run
        with tf.Session() as sess1:
            y_val, z_val = sess1.run([y, z])
        self.assertEqual(y_val, 8)
        self.assertEqual(z_val, 13)

    def test_linear_regression(self):
        """
        Linear regression, using nomal matrix.
        (theta = XT*X)-1*XT*y
        """
        from sklearn.datasets import fetch_california_housing
        housing = fetch_california_housing()
        m, n = housing.data.shape
        housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]

        X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name="X")
        y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
        XT = tf.transpose(X)
        theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)
        with tf.Session() as sess:
            theta_value = theta.eval()
            print(theta_value)

        
if __name__ == '__main__':
    unittest.main()


