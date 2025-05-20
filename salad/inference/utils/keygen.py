import jax

class KeyGen:
    def __init__(self, seed):
        self.key = jax.random.PRNGKey(seed)

    def __call__(self):
        self.key, key = jax.random.split(self.key)
        return key
