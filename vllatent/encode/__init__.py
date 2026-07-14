"""TORCH tier — frozen DINOv3 encoder wrappers.

Heavy imports (torch/timm) are LAZY (inside functions/methods) so a
torch-free box imports this package without crashing. Tested via `make test-torch`
on the dev box / H20, not in default CI.
"""
