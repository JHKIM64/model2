import keras
import tensorflow as tf
import keras.layers as layers

def get_trained_model(path) :
    location = {(37.6, 126.9), (37.6, 127.0), (37.5, 126.9), (37.5, 127.0), (37.5, 127.1), (37.4, 126.9), (37.4, 127.0),
                (37.4, 127.1)}
    reconstructed_model=list()
    model_num=0
    for lat,lon in location :
        model = keras.models.load_model(path+str(lat)+'_'+str(lon)+'_'+"model")
        reconstructed_model.append(model)
        model_num += 1
    return reconstructed_model

def ensemble(models) :

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():  ###gpu process
        inputs = keras.Input(shape=(55,))
        mods = []
        for model in models:
            mods.append(model(inputs))
        outputs = layers.average(mods)
        averagemodel = keras.Model(inputs=inputs, outputs=outputs)
        # averagemodel.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        # averagemodel.fit(x_train,y_train, epochs=100, batch_size=5, verbose=1)

    return averagemodel