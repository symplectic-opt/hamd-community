"""
HAMD Community Edition
======================
Hyper-Adaptive Momentum Dynamics — open research / evaluation release.

Noncommercial use only.  See LICENSE for terms.
Commercial licensing: grserb.research@gmail.com
"""

__version__ = "1.0.0"
__license__ = "Noncommercial Research/Evaluation"

from hamd.core.native_cubic_hamd import NativeCubicHAMD
from hamd.core.utils import eval_native, eval_cubic, load_instance

__all__ = ["NativeCubicHAMD", "eval_native", "eval_cubic", "load_instance"]
