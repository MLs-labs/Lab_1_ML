"""
Test functions for the scikit-learn Student Performance lab (rg_log.ipynb).

Do not modify this file - it is used to verify your implementation.
"""

import numpy as np

GREEN = "\033[92m"
RESET = "\033[0m"


def split_test(X_train, X_test, y_train_reg, y_test_reg, y_train_clf, y_test_clf):
    assert X_train is not None and X_test is not None, \
        "X_train / X_test is None - call train_test_split first"

    assert X_train.shape == (400, 3), f"Expected X_train.shape=(400, 3), got {X_train.shape}"
    assert X_test.shape == (100, 3), f"Expected X_test.shape=(100, 3), got {X_test.shape}"
    assert len(y_train_reg) == 400 and len(y_test_reg) == 100, \
        "y_train_reg / y_test_reg have the wrong length - did you split with test_size=0.2?"
    assert len(y_train_clf) == 400 and len(y_test_clf) == 100, \
        "y_train_clf / y_test_clf have the wrong length - did you split with test_size=0.2?"

    assert np.isclose(np.sum(X_test), 8216.279476120473, atol=1e-1), \
        "X_test does not match the expected values - did you use random_state=1?"
    assert np.isclose(np.sum(y_test_reg), 6715.873344708287, atol=1e-1), \
        "y_test_reg does not match the expected values - did you split X, y_reg and y_clf together?"

    print(f"{GREEN}All tests passed!{RESET}")


def linear_model_test(target):
    assert hasattr(target, "coef_"), \
        "The model has not been fit yet - call LinearRegression().fit(X_train, y_train_reg)"

    assert target.coef_.shape == (3,), f"Expected 3 coefficients, got {target.coef_.shape}"
    assert np.allclose(target.coef_, [4.60527365, 0.35565135, 6.94649054], atol=1e-2), \
        f"coef_ does not match the expected weights, got {target.coef_}"
    assert np.isclose(target.intercept_, 6.647978267691329, atol=1e-1), \
        f"intercept_ does not match the expected bias, got {target.intercept_}"

    print(f"{GREEN}All tests passed!{RESET}")


def logistic_model_test(target):
    assert hasattr(target, "coef_"), \
        "The model has not been fit yet - call LogisticRegression().fit(X_train, y_train_clf)"

    assert target.coef_.shape == (1, 3), f"Expected coef_.shape=(1, 3), got {target.coef_.shape}"
    assert np.allclose(target.coef_[0], [1.54586144, 0.11320178, 2.26889642], atol=1e-2), \
        f"coef_ does not match the expected weights, got {target.coef_}"
    assert np.isclose(target.intercept_[0], -14.26201978, atol=1e-1), \
        f"intercept_ does not match the expected bias, got {target.intercept_}"

    print(f"{GREEN}All tests passed!{RESET}")
