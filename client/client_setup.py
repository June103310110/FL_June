import tensorflow as tf
import secrets


class client_k():
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
        self.warmingUp = False
        
    def get_config(self):
        config = self.attri
        config['model'] = self.model
        config['optimizer'] = self.optimizer
        config['loss_fn'] = self.loss_fn
        config['warmingUp'] = self.warmingUp
        return self.attri
    
def generate_key():
    key = secrets.token_urlsafe(16)
    return key


