
import jax
import flax.linen as nn

import copy
from transformers.models.t5.modeling_flax_t5 import (
    FlaxT5Module,
    FlaxT5Stack,
    FlaxT5ForConditionalGeneration,
    FlaxT5ForConditionalGenerationModule
)



# Model hyperparameters, for convenience
OUTPUT_VOCAB_SIZE = 16384 + 1  # encoded image token space + 1 for bos
OUTPUT_LENGTH = 256 + 1  # number of encoded tokens + 1 for bos
BOS_TOKEN_ID = 16384
BASE_MODEL = 'Wikidepia/IndoT5-large' # we currently have issues with bart-large


class CustomT5Module(FlaxT5Module):
    def setup(self):
        self.config.vocab_size_output = getattr(self.config, 'vocab_size_output', OUTPUT_VOCAB_SIZE)
        self.config.max_position_embeddings_decoder = getattr(self.config, 'max_position_embeddings_decoder', OUTPUT_LENGTH)
        self.shared = nn.Embed(
            self.config.vocab_size,
            self.config.d_model,
            embedding_init=jax.nn.initializers.normal(self.config.initializer_factor * 1.0, self.dtype),
            dtype=self.dtype,
        )

        self.decoder_embed = nn.Embed(
            self.config.vocab_size_output,
            self.config.d_model,
            embedding_init=jax.nn.initializers.normal(self.config.initializer_factor * 1.0, self.dtype),
            dtype=self.dtype,
        )

        encoder_config = copy.deepcopy(self.config)
        encoder_config.causal = False
        self.encoder = FlaxT5Stack(encoder_config, embed_tokens=self.shared, dtype=self.dtype)

        decoder_config = copy.deepcopy(self.config)
        decoder_config.causal = True
        decoder_config.num_layers = self.config.num_decoder_layers
        decoder_config.max_position_embeddings = self.config.max_position_embeddings_decoder
        decoder_config.vocab_size = self.config.vocab_size_output
        self.decoder = FlaxT5Stack(decoder_config, embed_tokens=self.decoder_embed, dtype=self.dtype)

class CustomFlaxT5ForConditionalGenerationModule(FlaxT5ForConditionalGenerationModule):
    def setup(self):
        # check config is valid, otherwise set default values
        self.config.vocab_size_output = getattr(self.config, 'vocab_size_output', OUTPUT_VOCAB_SIZE)

        self.config.max_position_embeddings_decoder = getattr(self.config, 'max_position_embeddings_decoder', OUTPUT_LENGTH)
        self.shared = nn.Embed(
            self.config.vocab_size,
            self.config.d_model,
            embedding_init=jax.nn.initializers.normal(self.config.initializer_factor * 1.0, self.dtype),
            dtype=self.dtype,
        )

        self.decoder_embed = nn.Embed(
            self.config.vocab_size_output,
            self.config.d_model,
            embedding_init=jax.nn.initializers.normal(self.config.initializer_factor * 1.0, self.dtype),
            dtype=self.dtype,
        )

        encoder_config = copy.deepcopy(self.config)
        encoder_config.causal = False
        self.encoder = FlaxT5Stack(encoder_config, embed_tokens=self.shared, dtype=self.dtype)

        decoder_config = copy.deepcopy(self.config)
        decoder_config.causal = True
        decoder_config.num_layers = self.config.num_decoder_layers
        decoder_config.max_position_embeddings = self.config.max_position_embeddings_decoder
        decoder_config.vocab_size = self.config.vocab_size_output
        self.decoder = FlaxT5Stack(decoder_config, embed_tokens=self.decoder_embed, dtype=self.dtype)

        self.model = CustomT5Module(config=self.config, dtype=self.dtype)
        self.lm_head = nn.Dense(
            self.config.vocab_size_output,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_factor, self.dtype),
        )
        #self.final_logits_bias = self.param("final_logits_bias", self.bias_init, (1, self.config.vocab_size_output))

class CustomFlaxT5ForConditionalGeneration(FlaxT5ForConditionalGeneration):
    module_class = CustomFlaxT5ForConditionalGenerationModule
