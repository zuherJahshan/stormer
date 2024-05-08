import tensorflow as tf


class GatedMlpBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        inner_dim,
        outer_dim,
        non_linearity,
    ):
        super(GatedMlpBlock, self).__init__()
        self.inner_dim = inner_dim
        self.outer_dim = outer_dim
        self.inner_dense_non_linear = tf.keras.layers.Dense(
            units=inner_dim,
            activation=non_linearity,
        )
        self.inner_dense_linear = tf.keras.layers.Dense(
            units=inner_dim,
        )
        self.outer_dense = tf.keras.layers.Dense(
            units=outer_dim,
        )

    def call(self, input_seq):
        inner_non_linear = self.inner_dense_non_linear(input_seq)
        inner_linear = self.inner_dense_linear(input_seq)
        multiply = inner_non_linear * inner_linear
        return self.outer_dense(multiply)
    
    def get_mac(self, seq_len):
        return 3*seq_len*self.inner_dim*self.outer_dim + seq_len*self.inner_dim


class RotaryPositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, theta_0, projection_dim):
        super(RotaryPositionalEncoding, self).__init__()
        self.indices = tf.constant([(i // 2) for i in range(projection_dim)], dtype=tf.float32)
        self.thetas = theta_0 ** (-2 * (self.indices / projection_dim)) # thetas are of shape (projection_dim,)


    def call(self, input_seq):
        # input_seq is of shape (batch, input_seq_size, projection_dim)
        # compute the positional encoding
        input_seq_shape = tf.shape(input_seq)
        batch_size = input_seq_shape[0]
        input_seq_size = input_seq_shape[1]
        # create a vector of indices
        seq_indices = tf.range(0, input_seq_size, 1, dtype=tf.float32) # indices are of shape (input_seq_size,)
        # we need to create a matrix of shape (input_seq_size, projection_dim)
        seq_indices = tf.expand_dims(seq_indices, axis=-1)
        seq_indices = tf.tile(seq_indices, [1, tf.shape(input_seq)[2]])
        linear_phase = seq_indices * self.thetas

        # calculate the phase with consnie
        phased_with_cos = input_seq * tf.math.cos(linear_phase)

        # Rotate and multiply by [-1,1,-1,1,...] to calculate the phase with sine
        shifted_input_seq = tf.reshape(input_seq, [batch_size, input_seq_size, -1, 2])
        shifted_input_seq = tf.roll(shifted_input_seq, shift=1, axis=-1)
        shifted_input_seq = shifted_input_seq * tf.constant([-1,1], dtype=tf.float32)
        shifted_input_seq = tf.reshape(shifted_input_seq, [batch_size, input_seq_size, -1])
        phased_with_sin =  tf.math.sin(linear_phase) * shifted_input_seq
        
        return phased_with_cos + phased_with_sin


    def get_mac(self, seq_len):
        return 0 # TODO: need to be fixed


class MultiQueryAttention(tf.keras.layers.Layer):
    def __init__(
        self,
        num_heads,
        proj_dim,
        dropout=0.0,
        kernel_regularizer=None,
    ):
        super(MultiQueryAttention, self).__init__()
        

        self.num_heads = num_heads
        self.proj_dim = proj_dim
        # define linear layers for key and value
        self.key_layer = tf.keras.layers.Dense(
            units=proj_dim,
            kernel_regularizer=kernel_regularizer,
        )
        self.value_layer = tf.keras.layers.Dense(
            units=proj_dim,
            kernel_regularizer=kernel_regularizer,
        )

        # define linear layers for query, as the number of heads
        self.query_layers = [tf.keras.layers.Dense(
            units=proj_dim,
            kernel_regularizer=kernel_regularizer,
        ) for _ in range(num_heads)]

        # define linear layer for output
        self.output_layer = tf.keras.layers.Dense(
            units=proj_dim,
            kernel_regularizer=kernel_regularizer,
        )


    def _compute_attn(
        self,
        query, # shape will be [B,S,d]
        input_keys, # shape will be [B,T,d]
        memory_keys, # shape will be [B,S,d]
        input_vals, # shape will be [B,T,d]
        memory_vals, # shape will be [B,S,d]
    ):
        # Assume S represents the number of memory cells and T represents the number of input cells
        # Compute the attention weights
        
        # Compute the score a memory cell gives to an input cell
        input_score = tf.matmul(query, input_keys, transpose_b=True)
        # Shape will be [B,S,T]. This will result in a matrix,
        # s.t. row i describes how much attention should the query i give all other input cells
        
        self_score = query * memory_keys
        self_score = tf.reduce_sum(self_score, axis=-1, keepdims=True)
        # Shape will be [B,S,1]. This will result in a vector,
        # s.t. element i describes how much attention should the query i give to itself

        # Concat self_score with input_score
        score = tf.concat([self_score, input_score], axis=-1)
        # Shape will be [B,S,T+1]. This will result in a matrix,
        # s.t. row i describes how much attention should the query i give to inputs and itself

        score /= tf.math.sqrt(tf.cast(tf.shape(input_keys)[-1], tf.float32))
        attn = tf.nn.softmax(score, axis=-1)

        # Break attn to [B,S,1] and [B,S,T]
        self_attn = attn[:, :, 0:1]
        input_attn = attn[:, :, 1:]

        value_of_input = tf.matmul(input_attn, input_vals) # shape will be [B,S,d]
        value_of_self = self_attn * memory_vals # shape will be [B,S,d]
        return value_of_input + value_of_self


    def call(self, input_seq, memory_cells):
        # query_seq is of shape (batch_size, input_size, key_dim)
        # store_seq is of shape (batch_size, store_seq, key_dim)
        # compute the attention weights
        ik = self.key_layer(input_seq)
        mk = self.key_layer(memory_cells)
        iv = self.value_layer(input_seq)
        mv = self.value_layer(memory_cells)
        attns = [self._compute_attn(q, ik, mk, iv, mv) for q in [layer(memory_cells) for layer in self.query_layers]]
        concat = tf.concat(attns, axis=-1)
        return self.output_layer(concat)
        

    def get_mac(self, seq_len, state_cells):
        return self.num_heads*state_cells*self.proj_dim**2 +\
            2*seq_len*self.proj_dim**2 +\
            2*self.num_heads*state_cells*seq_len*self.proj_dim +\
            state_cells*(self.proj_dim**2)*self.num_heads


class StateTransformerBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        num_heads,
        projection_dim,
        inner_ff_dim,
        dropout=0.0,
        kernel_regularizer=None,
    ):
        super(StateTransformerBlock, self).__init__()
        # primitive properties
        self.num_heads = num_heads
        self.projection_dim = projection_dim
        
        # layers
        self.attention = MultiQueryAttention(
            num_heads=num_heads,
            proj_dim=projection_dim,
            dropout=dropout,
            kernel_regularizer=kernel_regularizer,
        )
        self.add1 = tf.keras.layers.Add()
        self.layernorm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.gated_mlp = GatedMlpBlock(
            inner_dim=inner_ff_dim,
            outer_dim=projection_dim,
            non_linearity="relu",
        )
        self.ff_dropout = tf.keras.layers.Dropout(dropout)
        self.add2 = tf.keras.layers.Add()
        self.layernorm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)


    def call(self, state_seq, input_seq):
        # state sequence is of shape (batch_size, num_of_state_cells, projection_dim)
        # input sequence is of shape (batch_size, input_size, projection_dim)
        # store_seq = tf.concat([state_seq, input_seq], axis=1)
        attention_output = self.attention(input_seq, state_seq)
        attention_output = self.add1([attention_output, state_seq])
        attention_output = self.layernorm_1(attention_output)
        mlp_output = self.gated_mlp(attention_output)
        outer_output = self.ff_dropout(mlp_output)
        outer_output = self.add2([outer_output, attention_output])
        return self.layernorm_2(outer_output) # the output is of shape (batch_size, num_of_state_cells, projection_dim)
    

    def get_mac(self, seq_len, state_cells):
        return self.attention.get_mac(seq_len, state_cells) +\
            self.gated_mlp.get_mac(state_cells)
    

class StormerRU(tf.keras.layers.Layer):
    def __init__(
        self,
        num_heads,
        num_state_cells,
        projection_dim,
        inner_ff_dim,
        initial_state_trainability=False,
        dropout=0.0,
        kernel_regularizer=None,
    ):
        super(StormerRU, self).__init__()

        self.num_state_cells = num_state_cells

        # Initialize the learnable initial state
        self.initial_state = self.add_weight(
            shape=(1, num_state_cells, projection_dim),
            initializer='random_normal',
            trainable=True,
            name='initial_state'
        )

        # State TE layers
        self.calc_z = StateTransformerBlock(
            num_heads=num_heads,
            projection_dim=projection_dim,
            inner_ff_dim=inner_ff_dim,
            dropout=dropout,
            kernel_regularizer=kernel_regularizer,
        )
        self.calc_r = StateTransformerBlock(
            num_heads=num_heads,
            projection_dim=projection_dim,
            inner_ff_dim=inner_ff_dim,
            dropout=dropout,
            kernel_regularizer=kernel_regularizer,
        )
        self.calc_current_state = StateTransformerBlock(
            num_heads=num_heads,
            projection_dim=projection_dim,
            inner_ff_dim=inner_ff_dim,
            dropout=dropout,
            kernel_regularizer=kernel_regularizer,
        )
        
        self.passthrough = StateTransformerBlock(
            num_heads=num_heads,
            projection_dim=projection_dim,
            inner_ff_dim=inner_ff_dim,
            dropout=dropout,
            kernel_regularizer=kernel_regularizer,
        )

        self.residual_add = tf.keras.layers.Add()
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)


    def set_initial_state_trainability(self, trainable):
        self.initial_state._trainable = trainable


    def call(self, input_seq):
        # Assume that input is of size [B,T,S,D] where B is the batch size, T is the number of time steps, S is the sequence length at each timestep, and D is the feature dimension
        # initialize the state sequence
        batch_size = tf.shape(input_seq)[0]
        # Use the learnable initial state, replicate it for the whole batch
        state_t = tf.tile(self.initial_state, [batch_size, 1, 1])

        folds = tf.shape(input_seq)[1]
        states = []

        for fold in range(4):
            curr_input_seq = tf.gather(input_seq, fold, axis=1)
            # Calcualate the GRU-like gate output
            z = self.calc_z(state_t, curr_input_seq)
            r = self.calc_r(state_t, curr_input_seq)
            current_state = self.calc_current_state(r * state_t, curr_input_seq)
            state_t = (1 - z) * state_t + z * current_state
            
            # Add the gated input.
            passthrough = self.passthrough(state_t, curr_input_seq)
            state_t = self.residual_add([state_t, passthrough])
            state_t = self.norm(state_t)
            states.append(state_t)

        # change the dimensions 
        return tf.transpose(tf.stack(states), [1, 0, 2, 3])



    def get_mac(self, seq_len):
        return self.calc_z.get_mac(seq_len, self.num_state_cells) +\
            self.calc_r.get_mac(seq_len, self.num_state_cells) +\
            self.calc_current_state.get_mac(seq_len, self.num_state_cells)

class Stormer(tf.keras.models.Model):
    def __init__(
        self,
        num_classes,
        num_heads,
        num_repeats,
        num_state_cells,
        input_seq_size,
        projection_dim,
        inner_ff_dim,
        weight_decay,
        initial_state_trainability=False,
        dropout=0.0,
        **kwargs
    ):
        super(Stormer, self).__init__()
        # the input sequence size
        self.input_seq_size = input_seq_size
        self.num_state_cells = num_state_cells
        kernel_regularizer = tf.keras.regularizers.l2(weight_decay)

        # ITS recurrent units
        self.encoding = tf.keras.layers.Dense(
            units=projection_dim,
            kernel_regularizer=kernel_regularizer,
        )

        self.rope = RotaryPositionalEncoding(
            theta_0=10000,
            projection_dim=projection_dim,
        )
        
        self.itsrus = [ StormerRU(
            num_heads=num_heads,
            num_state_cells=num_state_cells[i],
            projection_dim=projection_dim,
            inner_ff_dim=inner_ff_dim,
            initial_state_trainability=initial_state_trainability,
            dropout=dropout,
            kernel_regularizer=kernel_regularizer,
        ) for i in range(num_repeats) ]

        self.classifier = tf.keras.layers.Dense(
            units=num_classes,
            activation="softmax",
        )


    def call(self, input_seq):
        # input_seq is of shape (batch_size, input_size, feature_dim).
        # First of all, we will transform it to the shape (batch_size, folds, input_seq_size, projection_dim)
        # Pad the input sequence to the nearest multiple of input_seq_size
        input_seq = self.encoding(input_seq)
        input_seq_size = input_seq.shape[1]
        folds = tf.cast(tf.math.ceil(input_seq_size / self.input_seq_size), tf.int32)
        final_time_steps = folds * self.input_seq_size
        input_seq = tf.pad(
            input_seq,
            [[0, 0], [0, final_time_steps - input_seq_size], [0, 0]]
        )
        input_seq = self.rope(input_seq)
        
        input_seq = tf.reshape(
            input_seq,
            [-1, folds, self.input_seq_size, input_seq.shape[-1]]
        )
        # pass the input sequence through the ITSRUs
        x = input_seq
        for itsru in self.itsrus:
            x = itsru(x)

        return self.classifier(x[:, -1, 0, :])
    

    def get_mac(self):
        return self.encoding.count_params() +\
            self.rope.get_mac(self.input_seq_size) +\
            self.itsrus[0].get_mac(self.input_seq_size) +\
            sum([itsru.get_mac(self.num_state_cells[i+1]) for i, itsru in enumerate(self.itsrus[1:])])