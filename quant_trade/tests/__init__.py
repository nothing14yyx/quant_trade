import os

# Expose repository-level tests package for internal imports
base_tests_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'tests')
)
package_tests_path = os.path.abspath(os.path.dirname(__file__))
__path__ = [base_tests_path, package_tests_path]
