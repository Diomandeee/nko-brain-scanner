"""
Minimal pytest compatibility shim for running pytest-style tests under unittest.
Provides a fixture decorator with dependency injection and a main() that
discovers and runs plain pytest-style test classes via introspection.
"""

import tempfile
import os
import sys
import inspect
import traceback

# Registry of fixtures
_fixtures = {}


class fixture:
    """Minimal pytest.fixture decorator replacement."""
    def __init__(self, func=None, **kwargs):
        if func is not None:
            self._func = func
            _fixtures[func.__name__] = func
        else:
            self._func = None
            
    def __call__(self, func=None):
        if func is not None:
            self._func = func
            _fixtures[func.__name__] = func
            return func
        if self._func is not None:
            return self._func()
        raise TypeError("fixture not properly initialized")


class approx:
    """Minimal pytest.approx replacement for floating-point comparison."""
    def __init__(self, expected, rel=None, abs=None):
        self.expected = expected
        self.rel_tol = rel if rel is not None else 1e-6
        self.abs_tol = abs if abs is not None else 1e-12

    def __eq__(self, other):
        import math
        if isinstance(self.expected, (int, float)):
            return math.isclose(other, self.expected, rel_tol=self.rel_tol, abs_tol=self.abs_tol)
        return other == self.expected

    def __repr__(self):
        return f"approx({self.expected})"


def _resolve_fixture(name, resolved_cache):
    """Resolve a fixture by name, handling dependencies and generators."""
    if name in resolved_cache:
        return resolved_cache[name]
    if name not in _fixtures:
        raise ValueError(f"Unknown fixture: {name}")
    
    func = _fixtures[name]
    # Get fixture's own parameters (excluding 'self')
    sig = inspect.signature(func)
    params = [p for p in sig.parameters if p != "self"]
    
    # Resolve dependencies first
    kwargs = {}
    for param in params:
        kwargs[param] = _resolve_fixture(param, resolved_cache)
    
    result = func(**kwargs)
    
    # Handle generator fixtures (yield-based)
    if inspect.isgenerator(result):
        value = next(result)
        resolved_cache[name] = value
        resolved_cache[f"_gen_{name}"] = result  # save for teardown
        return value
    
    resolved_cache[name] = result
    return result


def _teardown_fixtures(resolved_cache):
    """Run teardown for generator fixtures."""
    for key, gen in list(resolved_cache.items()):
        if key.startswith("_gen_") and inspect.isgenerator(gen):
            try:
                next(gen)
            except StopIteration:
                pass


def main(args=None, **kwargs):
    """Run pytest-style test classes found in the calling module."""
    # Get the caller's module globals
    frame = inspect.stack()[1]
    caller_globals = frame[0].f_globals

    passed = 0
    failed = 0

    for name, obj in sorted(caller_globals.items()):
        if not (isinstance(obj, type) and name.startswith("Test")):
            continue
        for method_name in sorted(dir(obj)):
            if not method_name.startswith("test_"):
                continue
            resolved_cache = {}
            try:
                instance = obj()
                if hasattr(instance, "setup_method"):
                    instance.setup_method()

                # Resolve fixture parameters for this test method
                method = getattr(instance, method_name)
                sig = inspect.signature(method)
                params = [p for p in sig.parameters if p != "self"]
                
                fixture_kwargs = {}
                for param in params:
                    fixture_kwargs[param] = _resolve_fixture(param, resolved_cache)
                
                method(**fixture_kwargs)
                
                if hasattr(instance, "teardown_method"):
                    instance.teardown_method()
                passed += 1
            except AssertionError:
                failed += 1
                print(f"FAIL {name}.{method_name}")
                traceback.print_exc()
            except Exception:
                failed += 1
                print(f"ERROR {name}.{method_name}")
                traceback.print_exc()
            finally:
                _teardown_fixtures(resolved_cache)

    total = passed + failed
    print(f"\n{'=' * 50}")
    print(f"Results: {passed} passed, {failed} failed, {total} total")
    if failed == 0:
        print("All tests passed!")
    print(f"{'=' * 50}")

    if failed > 0:
        sys.exit(1)
