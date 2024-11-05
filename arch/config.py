class Config:
    def __init__(self,
        vocab_size,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        input_layernorm=True,
        post_attention_layernorm=False,
        pre_ffnn_layernorm=True,
        post_ffnn_layernorm=False,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=None,
        rope_scaling=None,
        attention_dropout=0.0,
        attention_scale=0,
        embedding_multiplier=1.0,
        logits_scaling=1.0,
        residual_multiplier=1.0,
        attention_multiplier=1.0,
        attention_cap=None,
        logit_cap=None,
        attention_type="gqa",
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_dropout = attention_dropout
        self.attention_scale = attention_scale
        self.attention_type = attention_type

        self.input_layernorm = input_layernorm
        self.post_attention_layernorm = post_attention_layernorm
        self.pre_ffnn_layernorm = pre_ffnn_layernorm
        self.post_ffnn_layernorm = post_ffnn_layernorm


        self.embedding_multiplier = embedding_multiplier
        self.logits_scaling = logits_scaling
        self.residual_multiplier = residual_multiplier
        self.attention_multiplier = attention_multiplier
        self.attention_cap = attention_cap
        self.logit_cap = logit_cap

        self.tie_word_embeddings = tie_word_embeddings

        