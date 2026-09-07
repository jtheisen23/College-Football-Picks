"""College football predictions and betting-edge engine."""

import warnings

# The Python that ships with macOS links against LibreSSL, and urllib3 v2
# prints a warning about that the moment it is imported. It is harmless --
# certificates still verify -- but it goes to stderr on every invocation,
# which would mail the user the same notice every time a scheduled
# `cfbpicks snapshot` runs.
#
# The filter is registered here, in the package root, because the warning
# fires during `import urllib3` itself: by the time any submodule has
# imported requests it is already too late. Matching on the message rather
# than the exception class avoids importing urllib3 to reference it, which
# would trigger the very warning being suppressed.
warnings.filterwarnings(
    "ignore",
    message=r"urllib3 v2 only supports OpenSSL",
    category=Warning,
)

__version__ = "0.1.0"

__all__ = ["__version__"]
