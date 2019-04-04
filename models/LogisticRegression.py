import numpy as np

class LogisticRegression:
    def __init__(self, num_features):
        self.num_features = num_features
        self.W = np.zeros((self.num_features, 1))

    def train(self, x, y, epochs, batch_size, lr, optim):
        final_loss = None   # loss of final epoch

        # Train should be done for 'epochs' times with minibatch size of 'batch size'
        # The function 'train' should return the loss of final epoch
        # Loss of an epoch is calculated as an average of minibatch losses

        # ========================= EDIT HERE ========================
        # Define h(x) for computation
        def h(w, x_i):
            return self._sigmoid(-np.dot(w.T, x_i))

        # Determining the number of mini batches
        batches = x.shape[0] // batch_size
        if x.shape[0] % batch_size:
            batches += 1

        for i in range(epochs):
            for j in range(batches):
                # Separating batch data for ease of implementation
                batch_x = x[j * batch_size:(j + 1) * batch_size]
                batch_y = y[j * batch_size:(j + 1) * batch_size]

                # n is the batch size, reason for not using the batch_size : at the end of the epoch, leftover data
                # size become smaller than the the batch_size
                n = batch_x.shape[0]

                # Initialize gradient as the same shape with the W.
                gradient = np.zeros(self.W.shape)

                # Calculating the gradient, gave a little twist to the loop for better performance.
                for l in range(n):
                    tmp = h(self.W, batch_x[l]) - batch_y[l]
                    for k in range(self.num_features):
                        gradient[k] += tmp * batch_x[l][k]

                # Updating W.
                self.W = optim.update(self.W, gradient, lr)

                # Calculating loss for this epoch.
                final_loss = 0
                for l in range(n):
                    final_loss -= batch_y[l] * h(self.W, batch_x[l]) \
                                  + (1 - batch_y[l]) * np.log(1 - h(self.W, batch_x[l]))

            print('{}th epoch - loss: {}'.format(i, final_loss))
        print(self.W)
        # ============================================================
        return final_loss

    def eval(self, x):
        threshold = 0.5
        pred = None

        # Evaluation Function
        # Given the input 'x', the function should return prediction for 'x'
        # The model predicts the label as 1 if the probability is greater or equal to 'threshold'
        # Otherwise, it predicts as 0

        # ========================= EDIT HERE ========================
        def h(w, x_i): return self._sigmoid(-np.dot(x_i, w))

        pred = h(self.W, x) > threshold  # True == 1 and False == 0
        # ============================================================

        return pred

    def _sigmoid(self, x):
        sigmoid = None

        # Sigmoid Function
        # The function returns the sigmoid of 'x'

        # ========================= EDIT HERE ========================
        sigmoid = 1 / (1 + np.exp(x))
        # ============================================================
        return sigmoid