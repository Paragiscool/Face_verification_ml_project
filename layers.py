# Custom L1 Distance layer module 
# WHY DO WE NEED THIS: its needed to load the custom model

# Import dependencies
import tensorflow as tf
from tensorflow.keras.layers import Layer

# Custom L1 Distance Layer from Jupyter 
class L1Dist(Layer):
    
    # Init method - inheritance
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
       
    # Magic happens here - similarity calculation
    def call(self, input_embedding, validation_embedding=None):
        if validation_embedding is None:
            input_embedding, validation_embedding = input_embedding
        if isinstance(input_embedding, list):
            input_embedding = input_embedding[0]
        if isinstance(validation_embedding, list):
            validation_embedding = validation_embedding[0]
        return tf.math.abs(input_embedding - validation_embedding)
