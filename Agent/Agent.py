import os
import numpy as np

from keras.optimizers import Adam
from keras.layers import Input, Dense, Embedding, PReLU, BatchNormalization, Conv1D
from keras.models import Model

from Environnement.Environnement import Environnement
from Agent.Models import bidirectional_gru_model
from PriorityExperienceReplay.PriorityExperienceReplay import Experience

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

# Keep cutoff to an even number, so both size
class Agent:
    def __init__(self, cutoff=9, from_save=False, batch_size=128, lr=5*10e-5, discrim_loss_limit=0.2):

        self.cutoff = cutoff
        self.environnement = Environnement(cutoff=cutoff, min_frequency_words=300000)
        self.vocab = self.environnement.different_words

        self.batch_size = batch_size
        self.discriminator_targets = np.array([0 for _ in range(self.batch_size)].extend(
            [1 for _ in range(self.batch_size)]))
        self.lr = lr

        self.model = self._build_model()

        self.discriminator = self._build_discriminator()
        self.discriminator_loss_limit = discrim_loss_limit
        self.discriminator_loss = []

        self.dataset_epoch = 0

        if from_save is True:
            self.model.load_weights('model')
            self.discriminator.load_weights('discriminator')


    def _build_model(self):

        state_input = Input(shape=(self.cutoff,))

        embedding = Embedding(self.vocab + 1, 60, input_length=self.cutoff)(state_input)
        model = bidirectional_gru_model(embedding, 150, depth=3)
        output = Dense(self.vocab)(model)

        actor = Model(inputs=state_input, outputs=output)
        actor.compile(optimizer=Adam(lr=self.lr),
                      loss='mse')
        actor.summary()

        return actor

    def _build_discriminator(self):

        state_input = Input(shape=(self.cutoff,))

        embedding = Embedding(self.vocab + 1, 60, input_length=self.cutoff)(state_input)
        model = bidirectional_gru_model(embedding, 100, depth=2)
        discriminator_output = Dense(1, activation='sigmoid')(model)

        discriminator = Model(inputs=state_input, outputs=discriminator_output)
        discriminator.compile(optimizer=Adam(lr=self.lr),
                      loss='binary_crossentropy')

        discriminator.summary()
        return discriminator

    def train(self, epoch):

        while np.mean(self.discriminator_loss[-20:]) >= self.discriminator_loss_limit:
            self.discriminator_loss.append(self.train_discriminator())

        e, total_frames = 0, 0
        while e <= epoch:
            for i in range(10):
                self.train_actor()

            if np.mean(self.discriminator_loss[-20:]) >= self.discriminator_loss_limit:


    def train_discriminator(self, evaluate=False):
        real_batch = self.environnement.query_state(self.batch_size)
        pred = self.model.predict(real_batch)
        real_val = real_batch[:, self.cutoff//2]
        pred[:, real_val] = 0
        action = np.argmax(pred, axis=1)
        fake_batch = real_batch.copy()
        fake_batch[:, self.cutoff//2] = action
        batch = fake_batch.append(real_batch)
        if evaluate is True:
            return self.discriminator.evaluate(batch, self.discriminator_targets, verbose=False)
        else:
            return self.discriminator.train_on_batch(batch, self.discriminator_targets)

    # We have a very special environnement were we can have a value for every single decision at no extra cost,
    # we will then try to optimise everything at once
    def train_actor(self):
        batch = self.environnement.query_state(self.batch_size)
        targets = self.discriminator.predict([batch[:, self.cutoff//2] + i + batch[:, self.cutoff//2 + 1:] for i in range(self.vocab)])

if __name__ == '__main__':
    agent = Agent(cutoff=9, from_save=False, batch_size=128)
    agent.train(epoch=5000)
