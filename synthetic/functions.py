""" 
Test functions

Author(s): Wei Chen (wchen459@gmail.com)
"""

import numpy as np
import tensorflow as tf


class Function(object):
    
    def __init__(self):
        pass
    
    def evaluate(self, data):
        x = tf.placeholder(tf.float32, shape=[None, self.dim])
        y = self.equation(x)
        with tf.Session() as sess:
            values = sess.run(y, feed_dict={x: data})
        return values


class VLMOP2(Function):
    '''
    Reference:
        Van Veldhuizen, D. A., & Lamont, G. B. (1999, February). 
        Multiobjective evolutionary algorithm test suites. 
        In Proceedings of the 1999 ACM symposium on Applied computing (pp. 351-357).
    '''
    def __init__(self):
        self.dim = 2
        self.n_obj = 2
        self.name = 'VLMOP2'
        
        x1 = np.linspace(-1/self.dim**.5, 1/self.dim**.5, 100)
        x2 = np.linspace(-1/self.dim**.5, 1/self.dim**.5, 100)
        self.pareto_x = np.vstack((x1, x2)).T
        self.pareto_y = self.evaluate(self.pareto_x)
    
    def equation(self, x):
        transl = 1 / np.sqrt(2)
        part1 = (x[:, 0] - transl) ** 2 + (x[:, 1] - transl) ** 2
        part2 = (x[:, 0] + transl) ** 2 + (x[:, 1] + transl) ** 2
        y1 = tf.exp(-1 * part1)
        y2 = tf.exp(-1 * part2)
        return tf.stack((y1, y2), axis=1)
    
    
#class OKA1(Function):
#    '''
#    Reference: 
#        Okabe, T., Jin, Y., Olhofer, M., & Sendhoff, B. (2004, September). 
#        On test functions for evolutionary multi-objective optimization. 
#        In International Conference on Parallel Problem Solving from Nature (pp. 792-802). 
#        Springer, Berlin, Heidelberg.
#    '''
#    def __init__(self):
#        self.dim = 2
#        self.n_obj = 2
#        self.name = 'OKA1'
    
    
class NKNO1(Function):
    '''
    Normalized KNO1
    Reference: 
        J. Knowles. ParEGO: A hybrid algorithm with on-line landscape approximation for 
        expensive multiobjective optimization problems. Technical Report TR-COMPSYSBIO-2004-01, 
        University of Manchester, UK, 2004. Available from http://dbk.ch.umist.ac.uk/knowles/pubs.html
    '''
    def __init__(self):
        self.dim = 2
        self.n_obj = 2
        self.name = 'NKNO1'
        
        x1 = np.linspace(-0.5+4.4116/3-1, 0.5, 100)
        x2 = 4.4116/3 - x1 - 1
        self.pareto_x = np.vstack((x1, x2)).T
        self.pareto_y = self.evaluate(self.pareto_x)
        
    def equation(self, x):
        x = 3*(x+.5)
        r = 9 - (3*tf.sin(5/2*(x[:,0]+x[:,1])**2) + 3*tf.sin(4*(x[:,0]+x[:,1])) + 5*tf.sin(2*(x[:,0]+x[:,1])+2))
        phi = np.pi/12*(x[:,0]-x[:,1]+3)
        y1 = r/20*tf.cos(phi)
        y2 = r/20*tf.sin(phi)
        return tf.stack((y1, y2), axis=1)
    

if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Qt5Agg')
    
    N = 1000
    xs = np.random.uniform(-.5, .5, size=(N,2))
    func_obj = NKNO1()
    ys = func_obj.evaluate(xs)
    
    plt.figure()
    plt.subplot(121)
    true_pfx = func_obj.pareto_x
    plt.plot(true_pfx[:,0], true_pfx[:,1])
    plt.subplot(122)
    plt.scatter(ys[:,0], ys[:,1])
    true_pf = func_obj.pareto_y
    plt.plot(true_pf[:,0], true_pf[:,1])
    plt.show()