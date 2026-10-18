"""Tests. After a run in which every test passed (none failed or skipped), print a final message."""
import atexit
import unittest
from unittest import signals


def _report() -> None:
    results = list(signals._results.keys())
    if results and all(r.wasSuccessful() and not r.skipped for r in results):
        print("程式可正常執行!")


atexit.register(_report)
