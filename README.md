# nn


Computation graphs, back propagation and neural networks, done simple.

## low-level
```python
from nn.low import *
```

Provides low-level primitives to create computation graphs, including placeholders, parameters, and basic arithmetic operations.

See `nn.examples.linear_regression` for the usage of computation graph primitives.

## medium-level
```python
from nn.medium import *
```

Provides higher level operations useful for neural networks, including activations, loss functions, normalization, etc., 
based on basic operation nodes provided by `nn.low`.

See `nn.examples.xor` for the usage of medium-level APIs.

## high-level
```python
from nn.high import *
```

Provides Keras-like high-level abstraction for building and training neural networks.
