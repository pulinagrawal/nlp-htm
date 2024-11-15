from keras import layers, models
import keras as keras
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
class SparseAutoencoder:
    def __init__(self, encoding_dim=374, input_shape=(784,)):
        self.encoding_dim = encoding_dim
        self.input_shape = input_shape
        self._build_model()

    def _build_model(self):
        input_img = layers.Input(shape=self.input_shape)
        encoded = layers.Dense(self.encoding_dim, activation='relu',
                               activity_regularizer=keras.regularizers.l1(10e-5))(input_img)
        decoded = layers.Dense(self.input_shape[0], activation='sigmoid')(encoded)
        
        self.autoencoder = models.Model(input_img, decoded)
        self.encoder = models.Model(input_img, encoded)
        
        encoded_input = layers.Input(shape=(self.encoding_dim,))
        decoder_layer = self.autoencoder.layers[-1]
        self.decoder = models.Model(encoded_input, decoder_layer(encoded_input))
        
        self.autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    def train(self, x_train, epochs=50, batch_size=256, validation_data=None):
        self.autoencoder.fit(x_train, x_train, epochs=epochs, batch_size=batch_size,
                             shuffle=True, validation_data=validation_data)
    
    def reconstruct(self, x):
        return self.autoencoder.predict(x)

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

# Load and preprocess data
mnist = keras.datasets.mnist
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))


# Train autoencoder and save the model
autoencoder = SparseAutoencoder()
autoencoder.train(x_train, epochs=50, batch_size=256, validation_data=(x_test, x_test))
# autoencoder.autoencoder.save('sparse_autoencoder_model.h5')

# # Load the model
# loaded_model = models.load_model('sparse_autoencoder_model.h5')

class AutoencoderSimpleGUI:
    def __init__(self, autoencoder: SparseAutoencoder, x_test):
        self.autoencoder = autoencoder
        self.x_test = x_test
        self.index = np.random.choice(len(self.x_test))
        self.fig, self.axes = plt.subplots(1, 2, figsize=(10, 4))
        self._display_single_image()
        self.index = 0

    def _display_single_image(self, event=None):
        self.index += 1
        original_img = self.x_test[self.index]
        reconstructed_img = self.autoencoder.predict(np.array([original_img]))[0]
        
        original_img = (original_img * 255).reshape(28, 28).astype(np.uint8)
        reconstructed_img = (reconstructed_img * 255).reshape(28, 28).astype(np.uint8)

        self.axes[0].imshow(original_img, cmap='gray')
        self.axes[0].set_title("Original")
        self.axes[0].axis('off')

        self.axes[1].imshow(reconstructed_img, cmap='gray')
        self.axes[1].set_title("Reconstructed")
        self.axes[1].axis('off')

        plt.draw()

    def run(self):
        ax_next = plt.axes([0.8, 0.01, 0.1, 0.075])
        btn_next = Button(ax_next, 'Next Image')
        btn_next.on_clicked(self._display_single_image)
        plt.show()

# Instantiate and run the simpler GUI with the loaded model
simple_gui = AutoencoderSimpleGUI(autoencoder.autoencoder, x_test)
simple_gui.run()