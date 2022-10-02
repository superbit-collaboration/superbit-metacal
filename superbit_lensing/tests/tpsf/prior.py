import numpy as np

class GaussianPrior(object):

    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def sample(self, sigma_clip=None):
        sample = np.random.normal(
            self.mu, self.sigma
            )

        if sigma_clip is not None:
            assert sigma_clip > 0
            chi = abs(sample - self.mu) / self.sigma
            if chi > sigma_clip:
                sample = self.sample(sigma_clip=sigma_clip)

        return sample

class UniformPrior(object):
    def __init__(self, left, right):
        self.left = left
        self.right = right

        return

    def sample(self):
        return np.random.uniform(self.left, self.right)
