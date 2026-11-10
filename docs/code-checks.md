# Code checks

Run the full regression suite before each focused commit:

```sh
.venv/bin/python -m pytest -q
```

Install development tools with `.venv/bin/python -m pip install -e '.[dev]'`.
A fresh environment also needs the `pilot` extra for exchange calendars.

Correctness lint for the automation and its service controls:

```sh
.venv/bin/ruff check tradingagents/pilot scripts/trade_automation.py --select F,E9,B
```

The broader repository style policy also reports existing one-line statements,
import placement/order and comprehension preferences in these modules. This
hardening change fixes correctness findings without reformatting the package.

Tests must redirect BASE, STATE and CONFIG to temporary paths and mock HTTP.
`tick(send=False)` suppresses delivery but still writes state. Never point a test
tick at production files. No service installation, restart or activation is part
of these checks.
