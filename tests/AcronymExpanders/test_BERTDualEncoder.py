'''
Created on Feb 9, 2019

@author: jpereira
'''
import unittest
import os

import tensorflow as tf
from tensorflow import keras

import bert

class Expander_BERTDualEncoder_Test(unittest.TestCase):

    """
        >>> mock = Mock()
    >>> cursor = mock.connection.cursor.return_value
    >>> cursor.execute.return_value = ['foo']
    >>> mock.connection.cursor().execute("SELECT 1")
    ['foo']
    """

    def build_model(self, bert_params):
        l_bert = bert.BertModelLayer.from_params(bert_params, name="bert")

        l_input_ids = keras.layers.Input(shape=(128,), dtype='int32', name="input_ids")
        l_token_type_ids = keras.layers.Input(shape=(128,), dtype='int32', name="token_type_ids")
        output = l_bert([l_input_ids, l_token_type_ids])
        output = keras.layers.Lambda(lambda x: x[:, 0, :])(output)
        output = keras.layers.Dense(2)(output)
        model = keras.Model(inputs=[l_input_ids, l_token_type_ids], outputs=output)

        model.build(input_shape=(None, 128))
        model.compile(optimizer=keras.optimizers.Adam(),
                      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")])

        for weight in l_bert.weights:
            print(weight.name)

        return model, l_bert

    def setUp(self) -> None:
        tf.compat.v1.reset_default_graph()
        tf.compat.v1.enable_eager_execution()
        print("Eager Execution:", tf.executing_eagerly())
        
        
    def test_albert_google_weights(self):
        albert_model_name = "albert_base"
        albert_dir = bert.fetch_tfhub_albert_model(albert_model_name, ".models")

        albert_params = bert.albert_params(albert_model_name)
        model, l_bert = self.build_model(albert_params)

        skipped_weight_value_tuples = bert.load_albert_weights(l_bert, albert_dir)
        self.assertEqual(0, len(skipped_weight_value_tuples))
        model.summary()
        
        
        
    def test_albert(self):
        bert_params = bert.BertModelLayer.Params(hidden_size=32,
                                                 vocab_size=67,
                                                 max_position_embeddings=64,
                                                 num_layers=1,
                                                 num_heads=1,
                                                 intermediate_size=4,
                                                 use_token_type=False,

                                                 embedding_size=16,  # using ALBERT instead of BERT
                                                 project_embeddings_with_bias=True,
                                                 shared_layer=True,
                                                 extra_tokens_vocab_size=3,
                                                 )


        def to_model(bert_params):
            l_bert = bert.BertModelLayer.from_params(bert_params)

            token_ids = keras.layers.Input(shape=(21,))
            seq_out = l_bert(token_ids)
            model = keras.Model(inputs=[token_ids], outputs=seq_out)

            model.build(input_shape=(None, 21))
            l_bert.apply_adapter_freeze()

            return model

        model = to_model(bert_params)
        model.summary()

        print("trainable_weights:", len(model.trainable_weights))
        for weight in model.trainable_weights:
            print(weight.name, weight.shape)
        self.assertEqual(23, len(model.trainable_weights))

        # adapter-ALBERT  :-)

        bert_params.adapter_size = 16

        model = to_model(bert_params)
        model.summary()

        print("trainable_weights:", len(model.trainable_weights))
        for weight in model.trainable_weights:
            print(weight.name, weight.shape)
        self.assertEqual(15, len(model.trainable_weights))

        print("non_trainable_weights:", len(model.non_trainable_weights))
        for weight in model.non_trainable_weights:
            print(weight.name, weight.shape)
        self.assertEqual(16, len(model.non_trainable_weights))

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()