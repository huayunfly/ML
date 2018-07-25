"""
Tiny example for regression ML.
@veison 2018.07.25
"""

import sys
import os
import unittest
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt


def error(f, x, y):
    return sp.sum((f(x) - y) ** 2)


class RegressionTest(unittest.TestCase):
    """
    Regression test cases.
    """

    def test_tiny_case(self):
        """
        Polyfit using the web traffic data, to predict the upper thredhold 100,000 access.
        We divide the data into 2 ranges.
        """
        data = sp.genfromtxt(
            os.path.join(sys.path[0], 'data/web_traffic.tsv'),
            delimiter='\t')
        x = data[:, 0]
        y = data[:, 1]
        x = x[~sp.isnan(y)]
        y = y[~sp.isnan(y)]
        plt.scatter(x, y, s=10)
        plt.title('Web traffic over the last month')
        plt.xlabel('Time')
        plt.ylabel('Hits/Hour')
        plt.xticks([w * 7 * 24 for w in range(10)],
                   ['week %i' % w for w in range(10)])
        plt.autoscale(tight=True)
        plt.grid(True, linestyle='-', color='0.75')

        # Polyfit in 2 ranges
        inflection = int(3.5 * 7 * 24)
        xa = x[:inflection]
        ya = y[:inflection]
        xb = x[inflection:]
        yb = y[inflection:]

        fp1 = sp.polyfit(xa, ya, 2)
        print('Model parameters fp1: %s' % fp1)
        fp2 = sp.polyfit(xb, yb, 2)
        print('Model parameters fp2: %s' % fp2)
        f1 = sp.poly1d(fp1)
        f2 = sp.poly1d(fp2)

        fx1 = sp.linspace(0, xa[-1], 1000)
        l_prev, = plt.plot(fx1, f1(fx1), linewidth=4,
                           label='3.5week prev order = %i, error = %f' % (
                               f1.order, error(f1, xa, ya))
                           )
        fx2 = sp.linspace(xb[0], xb[-1], 1000)
        l_after, = plt.plot(fx2, f2(fx2), linewidth=4,
                            label='3.5week after order = %i, error = %f' % (
                                f2.order, error(f2, xb, yb))
                            )
        plt.legend(handles=[l_prev, l_after], loc='upper left')
        plt.show()


if __name__ == '__main__':
    unittest.main()
