import unittest
import datetime
import numpy as np


class NumpyTest(unittest.TestCase):
    """
    Numpy usage and testing
    """

    def test_perf(self):
        arr = np.arange(1e7)
        larr = arr.tolist()
        scalar = 1.1

        begin = datetime.datetime.now()
        arr = arr * scalar
        end = datetime.datetime.now()
        print('numpy array times cost: %s' % (end - begin))

        begin = datetime.datetime.now()
        for i, val in enumerate(larr):
            larr[i] = val * scalar
        end = datetime.datetime.now()
        print('list array times cost: %s' % (end - begin))

    def test_create(self):
        alist = [1, 2, 3]
        arr = np.array(alist)
        self.assertEqual(arr.shape, (3,))

        arr1 = np.array([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(arr1.shape, (2, 3))

        arr2 = np.array([(1, 2, 3), (4, 5, 6)])
        self.assertEqual(arr2.shape, (2, 3))

        # arr1 == arr2 -> an element by element bool array
        self.assertEqual((arr1 == arr2).shape, (2, 3))
        self.assertTrue((arr1 == arr2)[1, 2])

        arr = np.zeros(3)
        self.assertEqual(arr.shape, (3,))

        arr = np.arange(0, 10, 1)
        self.assertEqual(arr.shape, (10,))

        arr = np.linspace(0, 1, 100)
        index = np.where(arr > 50)
        print(index)

        # Log10(start) to Log10(end) while in (start, stop)
        # in (0, 1) with 5 steps, including 1
        arr = np.logspace(0, 1, 5, 10)
        self.assertEqual(arr[4], 10)

    def test_datatype(self):
        # Designate data type
        cube = np.zeros((3, 3, 3)).astype(int) + 1
        self.assertEqual(cube.dtype, np.int)
        cube = np.ones((3, 3, 3)).astype(np.float16)
        self.assertEqual(cube.dtype, np.dtype('f2'))
        cube = np.zeros((3, 3, 3), dtype=np.float32)
        self.assertEqual(cube.dtype, np.dtype('f4'))

    def test_reshape(self):
        """
        The restructured arrays above are just different views of the same data in memory.
        This means that if you modify one of the arrays, it will modify the others.
        Uses numpy.copy() to separate arrays memory-wise
        """
        arr1d = np.arange(1000)
        arr3d = arr1d.reshape((10, 10, 10))
        self.assertEqual(arr3d.shape, (10, 10, 10))

        arr3d = np.reshape(arr1d, (10, 10, 10))
        self.assertEqual(arr3d.shape, (10, 10, 10))

        # Flatten
        arr4d = np.zeros((10, 10, 10, 10))
        arr1d = arr4d.ravel()
        self.assertEqual(arr1d.shape, (10000,))

    def test_record_1(self):
        # Creating an array of zeros and defining column types
        recarr = np.zeros((2,), dtype=('i4,f4,a10'))
        toadd = [(1, 2., 'Hello'), (2, 3., "World")]
        recarr[:] = toadd

    def test_record_2(self):
        # Creating an array of zeros and defining column types
        recarr = np.zeros((2,), dtype=('i4,f4,a10'))
        # Now creating the columns we want to put in the recarray
        col1 = np.arange(2) + 1
        col2 = np.arange(2, dtype=np.float32)
        col3 = ['Hello', 'World']
        toadd = list(zip(col1, col2, col3))
        recarr[:] = toadd

    def test_index(self):
        alist = [[1, 2], [3, 4], (5, 6)]
        arr = np.array(alist)
        self.assertEqual(arr.dtype, np.int64)
        self.assertEqual(arr[2, 1], 6)

    def test_bool_op(self):
        # Creating an image
        img1 = np.zeros((20, 20)) + 3
        img1[4:-4, 4:-4] = 6
        img1[7:-7, 7:-7] = 9
        # Let's filter out all values larger
        # than 2 and less than 6 or equal 9.
        index1 = img1 > 2
        index2 = img1 < 6
        index3 = img1 == 9
        index4 = (index1 & index2) | index3
        img3 = np.copy(img1)
        img3[index4] = 0
        self.assertEqual(img3[2, 2], 0)
        self.assertEqual(img3[10, 10], 0)

    def test_file_text(self):
        arr = np.array([(1, 2, 3), (4, 5, 6)])
        np.savetxt('text_file_text.txt', arr)

        arr = np.loadtxt('text_file_text.txt')
        self.assertEqual(arr[1, 2], 6)

    def test_file_bin(self):
        # Creating a large array to a binary file
        # numpy save and load (savez and loadz in zip mode)
        # format only with itself. Portable version is scipy.io
        data = np.empty((10, 10))
        np.save('test_file_bin.npy', data)
        newdata = np.load('test_file_bin.npy')
        self.assertEqual(newdata.shape, (10, 10))

    def test_math(self):
        # Using numpy.array is faster and less erroneous
        a = np.array([[3, 6, -5],
                      [1, -3, 2],
                      [5, -1, 4]])
        b = np.array([12, -2, 10])
        x = np.linalg.inv(a).dot(b)
        self.assertAlmostEqual(x[0], 1.75, delta=1e-6)
        self.assertAlmostEqual(x[1], 1.75, delta=1e-6)
        self.assertAlmostEqual(x[2], 0.75, delta=1e-6)

        # It is not wise to mix numpy.array and numpy.matrix
        # matrix is subclass of array
        A = np.matrix([[3, 6, -5],
                       [1, -3, 2],
                       [5, -1, 4]])
        B = np.matrix([[12],
                       [-2],
                       [10]])
        X = A ** (-1) * B
        self.assertAlmostEqual(X[0], 1.75, delta=1e-6)
        self.assertAlmostEqual(X[1], 1.75, delta=1e-6)
        self.assertAlmostEqual(X[2], 0.75, delta=1e-6)


if __name__ == '__main__':
    unittest.main()









