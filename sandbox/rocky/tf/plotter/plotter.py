import atexit
from queue import Empty,Queue
from rllab.sampler.utils import rollout
import numpy as np
from threading import Thread
import tensorflow as tf

__all__ = [
    'init_worker',
    'init_plot',
    'update_plot',
    'shutdown_worker'
]

process = None
queue = None

class PlotterThread(Thread):
    def __init__(self, queue,sess):
        super(PlotterThread,self).__init__()
        self.queue=queue
        self.sess = sess

    def run(self):
        env = None
        policy = None
        max_length = None
        while True:
            msgs = {}
            # Only fetch the last message of each type
            with self.sess.as_default():    
                with self.sess.graph.as_default():
                    while True:                    
                        try:
                            msg = self.queue.get_nowait()
                            msgs[msg[0]] = msg[1:]
                        except Empty:
                            break
                    if 'stop' in msgs:
                        break
                    elif 'update' in msgs:
                        env, policy = msgs['update']
                        # env.start_viewer()
                    elif 'demo' in msgs:            
                        param_values, max_length = msgs['demo']
                        policy.set_param_values(param_values)
                        rollout(env, policy, max_path_length=max_length, animated=True, speedup=5)                        
                    else:
                        if max_length:                            
                            rollout(env, policy, max_path_length=max_length, animated=True, speedup=5)


def shutdown_worker():
    if process:
        queue.put(['stop'])
        queue.task_done()
        queue.join()
        process.join()

def init_worker(sess=None):
    global queue, process
    queue = Queue()
    if sess is None:
        sess = tf.get_default_session()
    process = PlotterThread(queue, sess)
    process.start()

def init_plot(env, policy):
    queue.put(['update', env, policy])
    queue.task_done()

def update_plot(policy, max_length=np.inf):
    queue.put(['demo', policy.get_param_values(), max_length])
    queue.task_done()
