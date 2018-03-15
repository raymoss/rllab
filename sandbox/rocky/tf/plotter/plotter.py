import atexit
from queue import Empty, Queue
from threading import Thread

import numpy as np
import tensorflow as tf

from rllab.sampler.utils import rollout




__all__ = ['init_plot', 'update_plot']

thread = None
queue = None


class PlotterThread(Thread):
    def __init__(self, queue, sess):
        super(PlotterThread, self).__init__()
        self.queue = queue
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
                        if not self.sess._closed:
                            rollout(env, policy, max_path_length=max_length, animated=True, speedup=5)
                    else:
                        if max_length:
                            if not self.sess._closed:
                                rollout(env, policy, max_path_length=max_length, animated=True, speedup=5)


def _shutdown_worker():
    if thread:
        queue.put(['stop'])
        queue.task_done()
        queue.join()
        thread.join()


def _init_worker(**kwargs):
    global queue, thread
    if queue is None:
        queue = Queue()
        if 'sess' in kwargs:
            sess = kwargs.get('sess')
            if sess is None:
                sess = tf.get_default_session()
        else:
            sess = tf.get_default_session()
        thread = PlotterThread(queue, sess)
        thread.daemon = True
        thread.start()
        atexit.register(_shutdown_worker)


def init_plot(env, policy, **kwargs):
    _init_worker(**kwargs)
    queue.put(['update', env, policy])
    queue.task_done()


def update_plot(policy, max_length=np.inf):
    queue.put(['demo', policy.get_param_values(), max_length])
    queue.task_done()
