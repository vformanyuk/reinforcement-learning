import tensorflow as tf

class LearningRateDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate:float=1e-3, min_learning_rate:float=1e-5) -> None:
        super().__init__()
        self.min_learning_rate = min_learning_rate
        self.learning_rate = initial_learning_rate
        self.decay_step = (initial_learning_rate - self.min_learning_rate) / 200 #liner decay for 200 steps
    
    def decay(self):
        self.learning_rate = min(self.min_learning_rate, self.learning_rate - self.decay_step)

    def __call__(self, step):
        return self.learning_rate