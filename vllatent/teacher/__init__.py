"""TORCH tier — frozen WorldVLN teacher access.

The teacher runs as the UPSTREAM FastAPI server (``infer/run_server.sh`` -> uvicorn :8001,
USER-GATED: GPU + ~36.9 GB weights); this package is the HTTP **client** side — stdlib +
numpy, with the frame-PNG encoding lazily importing cv2/PIL. It never imports or modifies
the upstream clone. A torch-free box imports this package without crashing.
"""
