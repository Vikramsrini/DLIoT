"""
TrustEdge-CXR — Review 2 pipeline.

Each step is a module run from the project root, e.g.
    python -m trustedge.prepare_data
Nothing here imports TensorFlow or PyTorch at package level, so the
PyTorch teacher step and the TensorFlow student steps can run in
separate processes on the same Colab GPU.
"""
