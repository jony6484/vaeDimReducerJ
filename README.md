# vaeDimReducer
VAE Dim Reducer

This is a simple VAE dimension reducer,
it is fitted on a 2D array - either numpy or pytorch, and transforms
an array, from an input dim to a reduced dim.

# Example use:

```python 
# Loading a sample dataset
from sklearn.datasets import load_digits

X, y = load_digits(return_X_y=True)

# Reducing dim:
from VaeDimReducer.vaeDimReducer import VaeReducer
reducer = VaeReducer(input_dim=X.shape[-1], latent_dim=2, encoder_layers=[150]*2, decoder_layers=[150]*2)
# may work on either gpu or cpu, no need to verify inputs
reducer.to_gpu()
# Fitting:
reducer.fit(X, max_epochs=1500, target_r2=0.8, beta=0.2)
# traget_r2: the metric value for the vae reconstruction
# beta: the ratio between the KLD and MSE Losses - higher beta means more KLD
# max_batch_size: when using very large arrays, batches might be needed
# valid_size: default 0.2 - the size of input saved for validation
Z = reducer.transform(X)
```