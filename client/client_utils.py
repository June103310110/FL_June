import tensorflow as tf

class client():
    def __init__(self, name, dataset, model, 
                optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3),
                loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
                ):
        self.name = name
        self.dataset = dataset  
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        
        self.attri = {'name':self.name, 'dataset':self.dataset, 'optimizer': self.optimizer,
                     'loss_fn' : self.loss_fn}
        
        self.model.compile(self.optimizer, loss_fn,
                          metrics=metrics)
        
    def get_config(self):
        config = self.attri
        config['model'] = self.model
        config['optimizer'] = self.optimizer
        config['loss_fn'] = self.loss_fn
        return self.attri
