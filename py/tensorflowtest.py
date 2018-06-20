"""
Tensorflow test.

@author huayunfly@126.com
@date 2018.06.20
"""


import unittest
import datetime
import tensorflow as tf

class TensorflowTest(unittest.TestCase):
    """
    Basic test cases.
    """

    def test_a_graph(self):
        x = tf.Variable(3, name='x')
        y = tf.Variable(4, name='y')
        f = x*x*y + y + 2
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            init.run()
            result = f.eval()
        self.assertEqual(result, 42)

        with tf.Session() as sess1:
            sess1.run(x.initializer)
            sess1.run(y.initializer)
            result1 = sess1.run(f)
        self.assertEqual(result1, 42) 


if __name__ == '__main__':
    unittest.main()


