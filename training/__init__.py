"""Training scripts for the BioGraphX-Sol model variants.

This package exists so `inference.py` can import the PyTorch model class
definitions (GraphOnlyNN, ESMOnlyNN, GatedHybridNN) directly from
`train_nn_esol.py`, the script that trains them, avoiding a second copy of
the architectures.
"""
