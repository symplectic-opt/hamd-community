"""
HAMD Research Release
=====================
Hyper-Adaptive Momentum Dynamics — reduced-scope research implementation.

Noncommercial use only.  See LICENSE for terms.
"""

__version__ = "1.0.0"
__license__ = "Noncommercial Research/Evaluation"

from hamd.core.native_cubic_hamd import NativeCubicHAMD
from hamd.core.utils import eval_native, eval_cubic, load_instance

__all__ = ["NativeCubicHAMD", "eval_native", "eval_cubic", "load_instance"]
