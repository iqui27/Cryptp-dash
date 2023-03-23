from Dashboard import fetch_ohlcv_data,prepare_data, create_rnn_model


# Funções fetch_ohlcv_data, prepare_data e create_rnn_model

ohlcv_data = fetch_ohlcv_data()
X_train, X_test, y_train, y_test, scaler = prepare_data(ohlcv_data)

input_shape = (X_train.shape[1], X_train.shape[2])
model = create_rnn_model(input_shape)

model.fit(X_train, y_train, epochs=10, batch_size=32)

# Salve os pesos do modelo treinado
model.save_weights('rnn_model_weights.h5')