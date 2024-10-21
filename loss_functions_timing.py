# import timeit
import optax
import jax
import jax.numpy as jnp
import numpy as np
from sklearn.datasets import make_blobs


def jnp_svm_l1_objective(alpha, K, y):
    """SVM loss function.

    Parameters as np version.
    JAX version, is this faster?
    """
    return 0.5 * jnp.multiply(alpha, y).dot(K).dot(jnp.multiply(y, alpha)) - jnp.dot(alpha, jnp.ones(alpha.shape[0]))


def rbf_kernel(X1, X2, gamma=0.1):
    sqdist = (
        np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
    )
    return np.exp(-gamma * sqdist)


def trustful_bisection(fun, start, end, params, tol=1e-9, max_iter=100):
    """Bisection where fun(start) and fun(end) are trusted
    to have different signs.
    """
    f_start = fun(start, params)
    f_end = fun(end, params)
    direction = jnp.where(f_start < 0,
                          jnp.where(f_end > 0, 1, 0),
                          jnp.where(f_end < 0, -1, 0))

    # print(direction, f_start, f_end, start, end)
    def stopping_condition(state):
        return jnp.where(state[0] > tol,
                         jnp.where(state[4] < max_iter, True, False),
                         False)
        # return state[0] > tol

    def body_fun(state):
        start, end = state[1]
        direction = state[2]
        _iter = state[4]
        half = (start + end) * 0.5
        val_half = fun(half, params)
        _match = direction * val_half
        start = jnp.where(_match > 0, start, half)
        end = jnp.where(_match > 0, half, end)
        return (val_half**2, (start, end),  direction, half, _iter+1)

    init_state = (1, (start, end), direction, 0, 0)
    return jax.lax.cond(direction == 0,
                     lambda: jnp.where(f_start < f_end, start, end),
                     lambda: jax.lax.while_loop(stopping_condition, body_fun, init_state)[3])

@jax.jit
def svm_project(alpha, y, C):
    params = (C, y, alpha)
    def svm_bisection_optimality(t, paramss):
        C, y, alpha = paramss
        val = jnp.clip(alpha + y * t, 0, C).dot(y)
        return val

    # from jaxopt source
    lower = jax.lax.stop_gradient(jnp.min((-alpha) / y))
    upper = jax.lax.stop_gradient(jnp.max((-alpha + C) / y))
    val =  jnp.clip(alpha + y * trustful_bisection(svm_bisection_optimality, lower, upper, params), 0, C)
    # print(val.dot(y))
    return val

@jax.jit
def project_gradient(grad, jnp_alpha, jnp_y, c):
    pk = jnp_alpha - grad
    pk = svm_project(pk, jnp_y, c)
    return -(pk - jnp_alpha)

@jax.jit
def solve_svm(jnp_K, jnp_y, c, max_iter=3000, tol=1e-8, eta=1):
    jnp_alpha = svm_project(jnp.zeros(K.shape[0]), jnp_y, c)
    
    transformer = optax.chain(
        optax.sgd(learning_rate=eta),
        optax.scale_by_backtracking_linesearch(
            max_backtracking_steps=50,  store_grad=True
        )
    )
    value_and_grad = jax.jit(optax.value_and_grad_from_state(jnp_svm_l1_objective))
    opt_state = transformer.init(jnp_alpha)

    first_val, _ = value_and_grad(jnp_alpha, jnp_K, jnp_y, state=opt_state)
    iter_state = (0, jnp_alpha, 1, opt_state, first_val)

    def condition(state):
        return jnp.where(state[0] < max_iter,
                         jnp.where(state[2] > tol,
                                   True,
                                   False),
                         False)

    upd = jax.jit(transformer.update, static_argnames=["value_fn"])
    appl = jax.jit(optax.apply_updates)
    def body(state):
        _iter, jnp_alpha, diff, opt_state, val_old = state
        value, grad = value_and_grad(jnp_alpha, jnp_K, jnp_y, state=opt_state)
        grad = project_gradient(grad, jnp_alpha, jnp_y, c)
        updates, opt_state = upd(
            grad, opt_state, jnp_alpha, value=value, grad=grad, value_fn=jnp_svm_l1_objective, K=jnp_K, y=jnp_y
        )
        jnp_alpha = appl(jnp_alpha, updates)
        jnp_alpha = svm_project(jnp_alpha, jnp_y, c)
        val_new = jnp_svm_l1_objective(jnp_alpha, jnp_K, jnp_y)
        return (_iter+1, jnp_alpha, ((val_old - val_new))**2, opt_state, val_new)

    res = jax.lax.while_loop(condition, body, iter_state)
    return res[0], res[1]


if __name__ == "__main__":
    # Create some example data
    n_samples = 2000
    X, y = make_blobs(n_samples=n_samples, n_features=2, centers=2, random_state=5)
    y = np.where(y == 1, 1, -1)
    # K = rbf_kernel(X, X)
    K = X @ X.T

    c = 0.1
    jnp_K = jnp.array(K)
    jnp_y = jnp.array(y)
    i, jnp_alpha = solve_svm(jnp_K, jnp_y, c, tol=1e-6, eta=1)
    print("took ", i, " iterations")
    ind = jnp_alpha > 1e-5

    print(jnp.max(jnp_alpha), jnp.min(jnp_alpha))
    print(jnp_alpha)
    jnp_y = jnp_y[ind]
    jnp_K = jnp_K[ind, :][:, ind]
    jnp_alpha = jnp_alpha[ind]
    b = -jnp_y + jnp_K.dot(jnp.diag(jnp_y)).dot(jnp_alpha)
    b = jnp.median(b)

    # predict
    p = jnp_K.dot(jnp.diag(jnp_y)).dot(jnp_alpha) - b
    p = jnp.where(p > 0, 1, -1)
    print(jnp.sum(p == jnp_y) / jnp_y.shape[0])

    r = 2
    x_min, x_max = X[:, 0].min() - r, X[:, 0].max() + r
    y_min, y_max = X[:, 1].min() - r, X[:, 1].max() + r
    xx2, yy2 = np.meshgrid(np.arange(x_min, x_max, .2),
                           np.arange(y_min, y_max, .2))
    x_pred = np.c_[xx2.ravel(), yy2.ravel()]

    kt = x_pred @ X[ind, :].T
    # kt = rbf_kernel(x_pred, X[ind, :])
    Z = np.where((kt @ jnp.diag(jnp_y)  @ (jnp_alpha) - b) > 0, 1, 0)
    Z = Z.reshape(xx2.shape)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.contourf(xx2, yy2, Z, cmap=plt.cm.coolwarm, alpha=0.3)
    ax.scatter(X[1:500, 0], X[1:500, 1], c=y[1:500], cmap=plt.cm.coolwarm, s=25)
    ax.scatter(X[ind, 0][1:500], X[ind, 1][1:500], c="y", s=25, marker="x")
    # ax.plot(xx,yy)
    ax.axis([x_min, x_max,y_min, y_max])
    plt.show()
