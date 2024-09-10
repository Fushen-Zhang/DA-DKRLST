import collections
import tensorflow as tf
import numpy as np
from tensorflow.python.util.tf_export import tf_export
from tensorflow.python.ops.rnn_cell_impl import LayerRNNCell
from tensorflow.python.layers import base
from tensorflow.python.framework import function
from tensorflow.python.ops.parallel_for.gradients import jacobian, batch_jacobian
from tensorflow.python.ops import math_ops

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"

@function.Defun()
def derivative(x, current_gradient):
    grad = 1. - (tf.nn.tanh(x) * tf.nn.tanh(x))
    return current_gradient * grad

@function.Defun(grad_func=derivative)
def StepFunction(x):
    return tf.nn.relu(tf.sign(x))

def sigmoid_grad(x):
    return math_ops.sigmoid(x) * (1 - math_ops.sigmoid(x))
def tanh_grad(x):
    return 1 - math_ops.tanh(x) * math_ops.tanh(x)
def identity_grad(x):
    return tf.ones_like(x)
def relu_grad(x):
    return tf.where(x >= 0, tf.ones_like(x), tf.zeros_like(x))

_DKRLSTTuple = collections.namedtuple("DKRLSTTuple", ("Dict", "Mu", "Sigma", "Q", "variance","m", "ew", "eb")) 
@tf_export("nn.rnn_cell.DKRLSTTuple")
class DKRLSTTuple(_DKRLSTTuple):
  __slots__ = ()
  @property
  def dtype(self):
    (ew,eb) = self
    if ew.dtype != eb.dtype:
       raise TypeError("Inconsistent internal state: %s vs %s vs %s vs %s vs %s" % (str(ew.dtype), str(eb.dtype)))
    return ew.dtype

class DKRLST(LayerRNNCell): 
    def __init__(self, M, sn2, jitter, Lambda, initW, initB, Update, units=1, reuse=None, name=None):
        super(DKRLST, self).__init__(_reuse=reuse, name=name)
        self.M = tf.convert_to_tensor(M,dtype=tf.float32)
        self.initW = initW
        self.initB = initB 
        self._num_units = units
        self._state_size =  ((1, None), (None, None),(None, None),(1, None),(None,1))
        self.M = tf.convert_to_tensor(M,dtype=tf.float32)
        self.sn2 = tf.convert_to_tensor(sn2,dtype=tf.float32)  
        self.jitter = tf.convert_to_tensor(jitter,dtype=tf.float32)
        self.Lambda  = tf.convert_to_tensor(Lambda,dtype=tf.float32)   
        self.lr_inc = 0.0
        self.Update = Update
        
    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._num_units

    def zero_state(self, batch_size,dtype=tf.float32,ini_state=None):
        state = []
        name_list = ["Dict", "Mu", "Sigma", "Q", "variance","m", "ew", "eb"]
        for i in range(len(ini_state)):
            state.append(tf.convert_to_tensor(ini_state[i],dtype = dtype,name=name_list[i]))
        return DKRLSTTuple(*state)

    def grad(self, grad, state, inp, out, optimizer, apply): 
        op_list = []
        
        for key in self.eligibility_trace_dict:
            el = getattr(state, self.eligibility_trace_dict[key])
            if self._bias.name in key:
                m_grad = tf.einsum('bj,k->bj', grad, el) 
            else: 
                m_grad = tf.einsum('bj,k->bj', grad, el) 
      
            m_grad = tf.reduce_sum(m_grad, 0)

            if apply:
                mod_grad = m_grad + self.eligibility_trace_storage_dict[self.eligibility_trace_dict[key]]
            else:
                mod_grad = tf.zeros_like(m_grad)
                
            with tf.control_dependencies([mod_grad]):
                if self._kernel.name in key and self._kernel.trainable:
                    
                    if apply:
                        op_list.append(optimizer.apply_gradients(zip([self.lr_inc*mod_grad], [self._kernel]), finish=False))
                        op_list.append(tf.assign(self.eligibility_trace_storage_dict[self.eligibility_trace_dict[key]], tf.zeros_like(self.eligibility_trace_storage_dict[self.eligibility_trace_dict[key]])))
                    else:
                        op_list.append(tf.assign_add(self.eligibility_trace_storage_dict[self.eligibility_trace_dict[key]], m_grad))
                
                if self._bias.name in key and self._bias.trainable:
                    
                    if apply:
                        op_list.append(optimizer.apply_gradients(zip([self.lr_inc*mod_grad], [self._bias]), finish=False))
                        op_list.append(tf.assign(self.eligibility_trace_storage_dict[self.eligibility_trace_dict[key]], tf.zeros_like(self.eligibility_trace_storage_dict[self.eligibility_trace_dict[key]])))
                    else:
                        op_list.append(tf.assign_add(self.eligibility_trace_storage_dict[self.eligibility_trace_dict[key]], m_grad))
        
        return op_list
    
    def grad_v(self, grad, vars, state, optimizer, apply): 
        op_list = []
        return op_list
    
    def get_grad(self, grad, state, inp, out, apply): 
        return_list = []
        var_list = []
        op_list = []
        return return_list, var_list, op_list

    def build(self, inputs_shape):
        self.eligibility_trace_dict = {}
        self.eligibility_trace_storage_dict = {}
        self.output_storage_ta = tf.TensorArray(size=0, dtype=tf.float32, dynamic_size=True)
        add_name = ''
        
        if type(self.initW) != np.ndarray:
            self._kernel = self.add_variable('KRLST/kernel' + add_name, shape=[1], initializer=tf.constant_initializer(self.initW))
        else:
            self._kernel = self.add_variable('KRLST/kernel' + add_name, shape=[1], initializer=tf.constant_initializer(self.initW))
        self.eligibility_trace_dict.update({self._kernel.name: 'ew'})
        self.el_kernel_storage = self.add_variable('KRLST/kernel_storage' + add_name, shape=[1], initializer=tf.zeros_initializer, trainable=False)
        self.eligibility_trace_storage_dict.update({'ew': self.el_kernel_storage})
        
        if type(self.initB) != np.ndarray:
            self._bias = self.add_variable('KRLST/bias' + add_name, shape=[1], initializer=tf.constant_initializer(np.random.normal(size=[1])))
        else:
            self._bias = self.add_variable('KRLST/bias' + add_name, shape=[1], initializer=tf.constant_initializer(self.initB))
        self.eligibility_trace_dict.update({self._bias.name: 'eb'})
        self.el_bias_storage = self.add_variable('KRLST/bias_storage' + add_name, shape=[1], initializer=tf.zeros_initializer, trainable=False)
        self.eligibility_trace_storage_dict.update({'eb': self.el_bias_storage})
        
        self.built = True

    def K(self,x1,x2,kernel,bias): 
        square_x1 = tf.reduce_sum(tf.einsum('bi,bi->bi',tf.einsum('bi,i->bi',x1, kernel),x1), axis=-1)  #tf.math.pow(tf.math.pow(kernel,2),-1)
        square_x2 = tf.reduce_sum(tf.einsum('bi,bi->bi',tf.einsum('bi,i->bi',x2,kernel),x2), axis=-1) 
        ex = tf.expand_dims(square_x1, axis=-1)
        ey = tf.expand_dims(square_x2, axis=-2)
        xy = tf.einsum('ij,kj->ik', tf.einsum('bi,i->bi',x1,  kernel ), x2)
        Re_dis = tf.square(bias)*tf.exp(-0.5*(ex - 2.0 * xy + ey))
        return Re_dis

    def criterion_1(self, m):
        return tf.concat([tf.ones([1,m[0]-1]),tf.zeros([1,1])],1)
    
    def criterion_2(self, Q, Mu):
        errors = tf.einsum('ij,jk->ik', Q, Mu)
        errors = tf.divide(errors, tf.linalg.tensor_diag_part(Q))
        criterion = tf.abs(errors)
        return criterion

    def criterion_3(self, m):
        return tf.ones([1,m[0]])

    def reduced_Update(self, Qold, Mu, Sigma, Dict,m):  
        m = m - 1
        return Qold, Mu[:-1,:], Sigma[:-1,:-1], Dict[:-1,:], m

    def delete_Update(self, Q, Mu, Sigma, Dict, r, m):
        Qs = tf.concat([Q[:r, r:r+1],Q[r+1:, r:r+1]],0)
        qs = Q[r,r]
        Ql = tf.concat([Q[:r,:r],Q[r+1:,:r]],0)
        Qr = tf.concat([Q[:r,r+1:],Q[r+1:,r+1:]],0)
        Q = tf.concat([Ql,Qr],1)
        Q = Q - tf.einsum('ij,kj->ik',Qs,Qs) * tf.pow(qs,-1)

        Mu = tf.concat([Mu[:r,:], Mu[r+1:,:]],0)

        Sigmal = tf.concat([Sigma[:r,:r],Sigma[r+1:,:r]],0)
        Sigmar = tf.concat([Sigma[:r,r+1:],Sigma[r+1:,r+1:]],0)
        Sigma = tf.concat([Sigmal,Sigmar],1)

        Dict = tf.concat([Dict[:r,:],Dict[r+1:,:]],0)
        m = m-1
        return Q, Mu, Sigma, Dict, m

    def no_Update(self, Q, Mu, Sigma, Dict, m):
        return Q, Mu, Sigma, Dict, m

    def loop1(self, input, state, target): 
        Dict, Mu, Sigma, Q, variance, m, ew, eb = state
 
        kt = self.K(input, Dict, self._kernel, self._bias)
        k = self.K(input, input, self._kernel, self._bias) + self.jitter
        q = tf.einsum('ij,kj->ik', Q, kt)
        y_mean = tf.einsum('ij,ik->jk', Mu, q)

        Gamma2 = k - tf.einsum('ij,jk->ik', kt, q)
        Gamma2 = tf.clip_by_value(Gamma2,0,1e3)
        h = tf.einsum('ij,jk->ik', Sigma, q)
        sf2 = Gamma2 + tf.einsum('ij,ik->jk', h, q)
        sf2 = tf.clip_by_value(sf2,0,1e3)
        sy2 = self.sn2 + sf2 
        variance = sy2

        Q = tf.pow(k,-1)
        Mu = target * k / (k + self.sn2)
        Sigma = k - tf.pow(k,2) / (k + self.sn2)
        Dict = tf.concat([Dict, input], 0)
        Dict = Dict[-1:,:]
        self.nums02ML = tf.pow(target,2)/(k+self.sn2)
        self.dens02ML = 1
        self.s02 = self.nums02ML / self.dens02ML
        m = tf.convert_to_tensor(np.ones((1)),dtype=np.float32)

        new_ew = tf.gradients(y_mean, self._kernel)[0] 
        new_eb = tf.gradients(y_mean, self._bias)[0] 

        return y_mean, DKRLSTTuple(Dict, Mu, Sigma, Q, variance, m, new_ew, new_eb) 

    def loop2(self, input, state, target):     
        Dict, Mu, Sigma, Q, variance, m, ew, eb = state   
        K = self.K(Dict, Dict, self._kernel, self._bias) 
        K = K + self.jitter * tf.eye(m[0])
        Sigma = self.Lambda * Sigma + (1-self.Lambda)*K 
        Mu = tf.sqrt(self.Lambda)*Mu
        Q, Mu, Dict = tf.stop_gradient(Q), tf.stop_gradient(Mu), tf.stop_gradient(Dict)

        kt = self.K(input, Dict, self._kernel, self._bias)
        ktt = self.K(input, input, self._kernel, self._bias) + self.jitter
        q = tf.einsum('ij,kj->ik', Q, kt)

        y_mean = tf.einsum('ij,ik->jk', Mu, q) 

        Gamma2 = ktt - tf.einsum('ij,jk->ik', kt, q)
        Gamma2 = tf.clip_by_value(Gamma2,0,1e3)
        h = tf.einsum('ij,jk->ik', Sigma, q)
        sf2 = Gamma2 + tf.einsum('ij,ik->jk', h, q)
        sf2 = tf.clip_by_value(sf2,0,1e3)
        sy2 = self.sn2 + sf2 

        if self.Update:
            Qold = Q
            p = tf.concat([q,-tf.ones([1,1])],0)
            Q = tf.concat([Q,tf.zeros([1,m[0]])],0)
            Q = tf.concat([Q,tf.zeros([m[0]+1,1])],1) + tf.pow(Gamma2,-1) * tf.einsum('ij,kj->ik',p,p)

            p = tf.concat([q,sf2],0)
            Mu = tf.concat([Mu, y_mean],0) + (target-y_mean)/sy2*p
            Sigma = tf.concat([Sigma,h],1)
            Sigma = tf.concat([Sigma,tf.concat([tf.transpose(h),sf2],1)],0)
            m = m + 1
            Dict = tf.concat([Dict, input], 0)

            self.nums02ML = self.nums02ML + self.Lambda*tf.pow((target-y_mean),2)/sy2
            self.dens02ML = self.dens02ML + self.Lambda
            self.s02 = self.nums02ML/self.dens02ML
            self.prune = False

            criterion = tf.cond(tf.logical_or(tf.greater(tf.reshape(m,[]),self.M),tf.greater(tf.reshape(-Gamma2,[]),-self.jitter)),
            lambda:tf.cond(tf.greater(tf.reshape(-Gamma2,[]),-self.jitter),
                lambda: self.criterion_1(m),
                lambda: self.criterion_2(Q, Mu)
            ),
            lambda: self.criterion_3(m)
            )

            Q, Mu, Sigma, Dict, m = tf.cond(tf.logical_or(tf.greater(tf.reshape(m,[]),self.M),tf.greater(tf.reshape(-Gamma2,[]),-self.jitter)),
            lambda:tf.cond(tf.reduce_all(tf.equal(tf.arg_min(criterion,1,output_type=tf.int32), tf.to_int32(m))),
                lambda: self.reduced_Update(Qold, Mu, Sigma, Dict, m),
                lambda: self.delete_Update(Q, Mu, Sigma, Dict, tf.arg_min(criterion,1,output_type=tf.int32)[0], m)
            ),
            lambda: self.no_Update(Q, Mu, Sigma, Dict, m)
            )

        variance = tf.concat([variance,sy2],0)

        new_ew = tf.gradients(y_mean, self._kernel)[0]
        new_eb = tf.gradients(y_mean, self._bias)[0] 
            
        return y_mean, DKRLSTTuple(Dict, Mu, Sigma, Q, variance, tf.to_float(m), new_ew, new_eb) 

    def call(self, input, state, target): 

        y_mean, new_state = tf.cond(tf.reduce_all(tf.equal(state[5], tf.zeros_like(state[5]))),
                        lambda: self.loop1(input, state, target), 
                        lambda: self.loop2(input, state, target), 
        )

        return y_mean, new_state