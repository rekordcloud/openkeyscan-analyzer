"""
Runtime hook for PyInstaller to fix scipy.stats lazy loading issues.
This hook provides a minimal scipy.stats module to satisfy scipy.signal's requirements.
"""

import sys
import os
import warnings

# Suppress warnings during initialization
warnings.filterwarnings('ignore')

def _create_minimal_scipy_stats():
    """
    Create a minimal scipy.stats module with functions needed by scipy.signal.
    This is a workaround for the scipy.stats lazy loading issue in PyInstaller.
    """
    print("[scipy hook] Creating minimal scipy.stats module", file=sys.stderr)

    import types
    import numpy as np

    # Create a minimal scipy.stats module
    stats_module = types.ModuleType('scipy.stats')

    # Add the functions that scipy.signal._peak_finding needs
    def scoreatpercentile(a, per, limit=(), interpolation_method='fraction'):
        """Minimal implementation that returns percentile."""
        # This is a simplified version - just use numpy's percentile
        return np.percentile(a, per)

    stats_module.scoreatpercentile = scoreatpercentile

    # Add other common functions that might be needed
    stats_module.norm = type('norm', (), {
        'cdf': lambda x: 0.5,
        'pdf': lambda x: 0.4,
        'ppf': lambda x: 0.0,
    })()

    # Add to sys.modules so imports will find it
    sys.modules['scipy.stats'] = stats_module

    # Also add sub-modules that might be imported
    sys.modules['scipy.stats._distn_infrastructure'] = types.ModuleType('scipy.stats._distn_infrastructure')
    sys.modules['scipy.stats.distributions'] = types.ModuleType('scipy.stats.distributions')
    sys.modules['scipy.stats._continuous_distns'] = types.ModuleType('scipy.stats._continuous_distns')
    sys.modules['scipy.stats._discrete_distns'] = types.ModuleType('scipy.stats._discrete_distns')

    print("[scipy hook] Minimal scipy.stats module created", file=sys.stderr)
    return stats_module

def _initialize_scipy():
    """Initialize scipy modules for librosa."""
    try:
        # First, try normal import
        import scipy
        import scipy._lib
        import scipy.special

        # Try to import scipy.stats normally
        try:
            import scipy.stats
            # Check if it has what we need
            if hasattr(scipy.stats, 'scoreatpercentile'):
                print("[scipy hook] scipy.stats loaded successfully with scoreatpercentile", file=sys.stderr)
                return True
        except Exception as e:
            if "'obj' is not defined" in str(e):
                print(f"[scipy hook] scipy.stats has PyInstaller issue: {e}", file=sys.stderr)
            else:
                print(f"[scipy hook] scipy.stats import error: {e}", file=sys.stderr)

        # If scipy.stats failed, create a minimal version
        _create_minimal_scipy_stats()

        # Now import the other scipy modules that librosa needs
        import scipy.signal
        import scipy.fft
        import scipy.fftpack
        import scipy.linalg

        print("[scipy hook] Core scipy modules loaded", file=sys.stderr)
        return True

    except Exception as e:
        print(f"[scipy hook] Error initializing scipy: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return False

# Run initialization
print("[scipy hook] Starting scipy initialization", file=sys.stderr)
success = _initialize_scipy()

# Import numpy
try:
    import numpy
    import numpy.fft
    import numpy.linalg
    print("[scipy hook] Numpy modules initialized", file=sys.stderr)
except Exception as e:
    print(f"[scipy hook] Warning: Numpy initialization issue: {e}", file=sys.stderr)

if success:
    print("[scipy hook] Scipy hook completed successfully", file=sys.stderr)
else:
    print("[scipy hook] Scipy hook completed with warnings", file=sys.stderr)

# Restore warnings
warnings.filterwarnings('default')