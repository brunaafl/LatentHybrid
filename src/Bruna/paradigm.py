from moabb.paradigms import MotorImagery


class MotorImagery_(MotorImagery):

    def __init__(self, metric='accuracy', **kwargs):
        super().__init__(**kwargs)
        self.metric = metric

    @property
    def scoring(self):
        return self.metric
