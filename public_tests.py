"""
Test functions for the Student Performance regression / logistic regression lab.

Do not modify this file - it is used to verify your implementation.
"""

import numpy as np

GREEN = "\033[92m"
RESET = "\033[0m"


def _predict_logistic(X, w, b):
    return 1 / (1 + np.exp(-(X @ w + b)))


def split_test(df, df_train, df_test, X_train, X_test):
    assert df_train is not None and df_test is not None, \
        "df_train / df_test is None - implement the split first"

    assert len(df_train) + len(df_test) == len(df), \
        f"train + test sizes should add up to {len(df)}, got {len(df_train)} + {len(df_test)}"

    expected_train = int(0.8 * len(df))
    assert len(df_train) == expected_train, \
        f"Expected {expected_train} training rows, got {len(df_train)}"
    assert len(df_test) == len(df) - expected_train, \
        f"Expected {len(df) - expected_train} test rows, got {len(df_test)}"

    assert set(df_train.index).isdisjoint(set(df_test.index)), \
        "df_train and df_test share rows - make sure you split, not duplicate, the data"

    assert X_train.shape == (400, 3) and X_test.shape == (100, 3), \
        f"Expected X_train.shape=(400, 3) and X_test.shape=(100, 3), got {X_train.shape} and {X_test.shape}"

    print(f"{GREEN}All tests passed!{RESET}")


def compute_cost_linear_test(target):
    X = np.array([[1., 2.], [2., 1.], [3., 4.]])
    y = np.array([5., 4., 10.])

    cost = target(X, y, np.array([1., 1.]), 0.)
    assert np.isclose(cost, 2.3333333333333335), \
        f"Wrong cost. Expected 2.3333333333333335, got {cost}"

    y_perfect = X @ np.array([2., 0.]) + 1.
    cost = target(X, y_perfect, np.array([2., 0.]), 1.)
    assert np.isclose(cost, 0.0), \
        f"Cost should be 0 when w, b perfectly fit the data, got {cost}"

    print(f"{GREEN}All tests passed!{RESET}")


def compute_gradient_linear_test(target):
    X = np.array([[1., 2.], [2., 1.], [3., 4.]])
    y = np.array([5., 4., 10.])

    dj_dw, dj_db = target(X, y, np.array([1., 1.]), 0.)
    assert np.allclose(dj_dw, [-4.33333333, -5.66666667]), \
        f"Wrong dj_dw. Expected [-4.33333333, -5.66666667], got {dj_dw}"
    assert np.isclose(dj_db, -2.0), \
        f"Wrong dj_db. Expected -2.0, got {dj_db}"

    print(f"{GREEN}All tests passed!{RESET}")


def gradient_descent_test(target, cost_function, gradient_function):
    # a tiny synthetic problem with a known exact solution: y = 3x + 2
    X = np.array([[0.], [1.], [2.], [3.], [4.]])
    y = 3 * X.flatten() + 2

    w, b, J_history = target(
        X, y, np.zeros(1), 0., cost_function, gradient_function, 0.1, 1000
    )

    assert np.isclose(w[0], 3.0, atol=1e-3), \
        f"gradient descent should recover w=3 on y=3x+2, got w={w}"
    assert np.isclose(b, 2.0, atol=1e-3), \
        f"gradient descent should recover b=2 on y=3x+2, got b={b}"
    assert J_history[-1] < 1e-6, \
        f"final cost should converge close to 0, got {J_history[-1]}"

    print(f"{GREEN}All tests passed!{RESET}")


def train_linear_test(w_final, b_final, J_hist_linear):
    assert w_final is not None and b_final is not None and J_hist_linear is not None, \
        "w_final / b_final / J_hist_linear is None - call gradient_descent first"

    assert np.allclose(w_final, [13.79727181, 5.52377847, 10.24032161], atol=1e-2), \
        f"w_final does not match the expected weights, got {w_final}"
    assert np.isclose(b_final, 70.609036248947, atol=1e-1), \
        f"b_final does not match the expected bias, got {b_final}"
    assert J_hist_linear[-1] < J_hist_linear[0], \
        "the cost should decrease over training"
    assert J_hist_linear[-1] < 15, \
        f"final training cost should be below 15, got {J_hist_linear[-1]}"

    print(f"{GREEN}All tests passed!{RESET}")


def compute_cost_logistic_test(target):
    X = np.array([[1., 2.], [2., 1.], [3., 4.], [-1., -2.]])
    y = np.array([1., 0., 1., 0.])

    cost = target(X, y, np.array([0.5, -0.5]), 0.1)
    assert np.isclose(cost, 0.9752516014402621), \
        f"Wrong cost. Expected 0.9752516014402621, got {cost}"

    cost = target(X, y, np.zeros(2), 0.)
    assert np.isclose(cost, np.log(2)), \
        f"Cost should equal ln(2) when w=0, b=0 (f_wb=0.5 everywhere), got {cost}"

    print(f"{GREEN}All tests passed!{RESET}")


def compute_gradient_logistic_test(target, gradient_descent=None, cost_function=None):
    X = np.array([[1., 2.], [2., 1.], [3., 4.], [-1., -2.]])
    y = np.array([1., 0., 1., 0.])

    dj_dw, dj_db = target(X, y, np.array([0.5, -0.5]), 0.1)
    assert np.allclose(dj_dw, [-0.43727358, -1.05944557]), \
        f"Wrong dj_dw. Expected [-0.43727358, -1.05944557], got {dj_dw}"
    assert np.isclose(dj_db, 0.023484323056671697), \
        f"Wrong dj_db. Expected 0.023484323056671697, got {dj_db}"

    if gradient_descent is not None and cost_function is not None:
        # a tiny, linearly separable synthetic problem
        X_syn = np.array([[-2.], [-1.], [1.], [2.], [3.]])
        y_syn = np.array([0., 0., 1., 1., 1.])
        w, b, J_history = gradient_descent(
            X_syn, y_syn, np.zeros(1), 0., cost_function, target, 0.5, 2000
        )
        preds = (_predict_logistic(X_syn, w, b) >= 0.5).astype(int)
        assert np.array_equal(preds, y_syn.astype(int)), \
            f"the model should correctly classify every synthetic example, got {preds}"
        assert J_history[-1] < 0.01, \
            f"the cost should converge close to 0, got {J_history[-1]}"

    print(f"{GREEN}All tests passed!{RESET}")


def train_logistic_test(w_final_clf, b_final_clf, J_hist_logistic, predict_logistic, X_test_norm, y_test_clf):
    assert w_final_clf is not None and b_final_clf is not None and J_hist_logistic is not None, \
        "w_final_clf / b_final_clf / J_hist_logistic is None - call gradient_descent first"

    assert np.allclose(w_final_clf, [4.1253711, 1.42710176, 3.11593417], atol=1e-2), \
        f"w_final_clf does not match the expected weights, got {w_final_clf}"
    assert np.isclose(b_final_clf, 6.116281374820598, atol=1e-1), \
        f"b_final_clf does not match the expected bias, got {b_final_clf}"
    assert J_hist_logistic[-1] < J_hist_logistic[0], \
        "the cost should decrease over training"

    probs = predict_logistic(X_test_norm, w_final_clf, b_final_clf)
    preds = (probs >= 0.5).astype(int)
    accuracy = np.mean(preds == y_test_clf)
    assert accuracy >= 0.90, \
        f"test accuracy should be at least 90%, got {accuracy * 100:.2f}%"

    print(f"{GREEN}All tests passed!{RESET}")
