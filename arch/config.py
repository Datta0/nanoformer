class Config:
    def __init__(self,
        vocab_size,
        hidden_size=1024,
        intermediate_size=4096,
        num_hidden_layers=8,
        num_attention_heads=8,
        num_key_value_heads=2,
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
        attn_bias=False,
        mlp_bias=False,
        gradient_checkpointing=True,
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
        else:
            assert num_attention_heads % num_key_value_heads == 0, "num_attention_heads must be a multiple of num_key_value_heads"
        
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

        self.attn_bias = attn_bias
        self.mlp_bias = mlp_bias

        self.use_ngpt = "ngpt" in attention_type
        if self.use_ngpt:
            if self.input_layernorm or self.post_attention_layernorm or self.pre_ffnn_layernorm or self.post_ffnn_layernorm:
                print("NGPT enabled. Disabling explicit norms like input_layernorm, post_attention_layernorm, pre_ffnn_layernorm, or post_ffnn_layernorm.")
                self.input_layernorm = self.post_attention_layernorm = self.pre_ffnn_layernorm = self.post_ffnn_layernorm = False
            self.attention_type = "gqa-ngpt" # For now we only have nGPT for GQA. We might experiment with normalising DiffTransformer, MLA later.
        
        if attention_type == "mla":
            if kwargs.get("rope_head_dim", None) is not None:
                self.rope_head_dim = kwargs.get("rope_head_dim", None)
            
            assert "kv_compressed" in kwargs, "kv_compressed must be specified for MLA. Specify the same using --kv_compressed=<>"
            assert "q_compressed" in kwargs, "q_compressed must be specified for MLA. Specify the same using --q_compressed=<>"
            self.kv_compressed = kwargs.get("kv_compressed", None)
            self.q_compressed = kwargs.get("q_compressed", None)
            assert self.q_compressed // self.kv_compressed == self.num_attention_heads // self.num_key_value_heads, f"the ratio of q_compressed ({self.q_compressed}) to kv_compressed ({self.kv_compressed}) {self.q_compressed // self.kv_compressed} must be equal to the ratio of num_attention_heads ({self.num_attention_heads}) to num_key_value_heads ({self.num_key_value_heads}) {self.num_attention_heads // self.num_key_value_heads}" 


        self.tie_word_embeddings = tie_word_embeddings
        self.gradient_checkpointing = gradient_checkpointing

        # Set any extra kwargs as attributes if not already present
        for k, v in kwargs.items():
            if not hasattr(self, k):
                setattr(self, k, v)

