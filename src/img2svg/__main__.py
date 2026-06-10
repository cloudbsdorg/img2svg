# img2svg - Allow ``python -m img2svg`` to invoke the CLI.
# Copyright (c) 2026, CloudBSD
# SPDX-License-Identifier: BSD-3-Clause
"""Allow ``python -m img2svg`` to invoke the CLI.

Equivalent to running the ``img2svg`` console script after install.
"""

from img2svg.cli import app

if __name__ == "__main__":
    app()
