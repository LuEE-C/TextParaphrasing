from keras.layers import CuDNNLSTM, Bidirectional, Dropout, BatchNormalization, PReLU

def bidirectional_gru_model(input_tensor, gru_cells, depth=2):
    x = Bidirectional(CuDNNLSTM(gru_cells, return_sequences=True))(input_tensor)
    x = Dropout(0.3)(x)
    x = PReLU()(x)
    for _ in range(depth - 1):
        x = Bidirectional(CuDNNLSTM(gru_cells, return_sequences=True))(model_input)
        x = Dropout(0.3)(x)
        x = PReLU()(x)
    x = Bidirectional(CuDNNLSTM(gru_cells, return_sequences=False))(model_input)
    x = Dropout(0.3)(x)
    x = PReLU()(x)
    return x