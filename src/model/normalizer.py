import numpy as np

ndarray = np.ndarray

# Abstract Normalization Strategy
class NormalizationStrategy:
    def normalize(self, data):
        raise NotImplementedError

    def denormalize(self, data):
        raise NotImplementedError

# Tanh Normalization Strategy
class TanhNormalization(NormalizationStrategy):
    """
    >>> normalizer = TanhNormalization()
    >>> data = np.array([1, 2, 3, 4, 5])
    >>> normalizer.normalize(data)
    array([0.76159416, 0.96402758, 0.99505475, 0.9993293 , 0.9999092 ])
    >>> normalizer.denormalize(normalizer.normalize(data))
    array([1., 2., 3., 4., 5.])
    """
    def normalize(self, data):
        return np.tanh(data)

    def denormalize(self, data):
        return np.arctanh(data)

# Log Normalization Strategy
class LogNormalization(NormalizationStrategy):
    def normalize(self, data):
        return np.log(data)

    def denormalize(self, data):
        return np.exp(data)

# Sigmoid Normalization Strategy
class SigmoidNormalization(NormalizationStrategy):
    """
    >>> normalizer = SigmoidNormalization()
    >>> data = np.array([1, 2, 3, 4, 5])
    >>> normalizer.normalize(data)
    array([0.73105858, 0.88079708, 0.95257413, 0.98201379, 0.99330715])
    >>> normalizer.denormalize(normalizer.normalize(data))
    array([1., 2., 3., 4., 5.])
    """
    def normalize(self, data):
        return 1 / (1 + np.exp(-data))

    def denormalize(self, data):
        return -np.log((1 / data) - 1)
    
# Linear Normalization Strategy
class LinearNormalization(NormalizationStrategy):
    """
    >>> normalizer = LinearNormalization(10.0)
    >>> data = np.array([1, 2, 3, 4, 5])
    >>> normalizer.normalize(data)
    array([10., 20., 30., 40., 50.])
    >>> normalizer.denormalize(normalizer.normalize(data))
    array([1., 2., 3., 4., 5.])
    """
    def __init__(self, scale) -> None:
        self.scale = scale

    def normalize(self, data):
        return data * self.scale
    
    def denormalize(self, data):
        return data / self.scale

# Max Normalization Strategy
class MaxNormalization(LinearNormalization):
    """
    >>> data = np.array([1, 2, 3, 4, 5])
    >>> normalizer = MaxNormalization(data)
    >>> normalizer.normalize(data)
    array([0.2, 0.4, 0.6, 0.8, 1. ])
    >>> normalizer.denormalize(normalizer.normalize(data))
    array([1., 2., 3., 4., 5.])
    """
    def __init__(self, data):
        self.max = np.max(np.abs(data))
        super(MaxNormalization, self).__init__(1. / self.max)

# Std Normalization Strategy
class StdNormalization(LinearNormalization):
    def __init__(self, data):
        self.std = np.std(data).item()
        super(StdNormalization, self).__init__(1. / self.std)


# Normalizer Class
class Normalizer():
    """
    Normalize and denormalize data with given strategy.
    ### parameters 
    - strategy: NormalizationStrategy
        * TanhNormalization()
        * SigmoidNormalization()
        * LinearNormalization(scale)
        * MaxNormalization(data)
        * StdNormalization(data)

    >>> normalizer = Normalizer(TanhNormalization())
    >>> data = np.array([1, 2, 3, 4, 5])
    >>> normalizer.norm(data)
    array([0.76159416, 0.96402758, 0.99505475, 0.9993293 , 0.9999092 ])
    >>> normalizer.denorm(normalizer.norm(data))
    array([1., 2., 3., 4., 5.])
    """
    def __init__(self, strategy: NormalizationStrategy):
        self.strategy = strategy

    def norm(self, data):
        return self.strategy.normalize(data)
    
    def denorm(self, data):
        return self.strategy.denormalize(data)
    
class LayeredNormalizer():
    """
    Normalize and denormalize data in order with given strategies.
    ### parameters 
    - strategis: list[NormalizationStrategy]
        * TanhNormalization()
        * SigmoidNormalization()
        * LinearNormalization(scale)
        * MaxNormalization(data)
        * StdNormalization(data)

    >>> data = np.array([1, 2, 3, 4, 5])
    >>> strategies = [MaxNormalization(data), LinearNormalization(10.0)]
    >>> normalizer =  LayeredNormalizer(strategies)
    >>> normalizer.norm(data)
    array([ 2.,  4.,  6.,  8., 10.])
    >>> normalizer.denorm(normalizer.norm(data))
    array([1., 2., 3., 4., 5.])
    """
    def __init__(self, strategies: list[NormalizationStrategy]) -> None:
        self.strategies = strategies

    def norm(self, data):
        for strategy in self.strategies:
            data = strategy.normalize(data)
        return data
    
    def denorm(self, data):
        for strategy in self.strategies[::-1]:
            data = strategy.denormalize(data)
        return data
    

if __name__ == '__main__': 
    import doctest
    doctest.testmod()
