import numpy as np
from typing import Tuple, Union
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from scipy.stats import norm
from scipy.optimize import minimize

def blackbox_optimize(
        args_history: np.ndarray,
        func_vals_history: np.ndarray
) -> Union[np.ndarray, str]:

    """
    Функция, которая по истории проверенных точек и значений blackbox функции в них возращает точку, которую следует
    проверить следующей или же строку "stop". Учтите случай, что должна выдавать функция, когда истории нет
    (args_history и func_vals_history это пустые arrays)

    Args:
        args_history: история аргументов (args_history.shape = (n, 10))
        func_vals_history: история значений функции в соответствующих аргументах
    Returns:
        Следующая точка (np.ndarray размера 10)
    """

    if len(args_history) <= 1:
        return np.array([1]*10)
    
    ### ФИТТИНГ ГАУССОВСКОГО ПРОЦЕССА ###
    kernel = RBF()
    GP = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
    GP.fit(args_history, func_vals_history)
    
    
    ### ТОЧКИ ДЛЯ ACQUISITION FUNCTION ###
    maxs = np.max(args_history, axis=0)
    mins = np.min(args_history, axis=0)
    stds = np.std(args_history, axis=0)
    
    x_random = np.zeros((3000, 10))
    for i in range(10):
        x_random[:, i] = np.linspace(mins[i] - stds[i], maxs[i] + stds[i], 3000)
        
    def ei_maximization(x, gp, best_value, xi=0.01):
        m, sigma = gp.predict(x, return_std=True)
        sigma = np.maximum(sigma, 1e-8)
        
        EI = (best_value - m - xi) * norm.cdf((best_value - m - xi)/sigma) + sigma * norm.pdf((best_value - m - xi)/sigma)
        EI[sigma == 1e-8] = 0 
        return EI
    
    best_value = np.min(func_vals_history)
    EI = ei_maximization(x_random, GP, best_value)
    top_50_indices = np.argsort(EI)[::-1][:50]
    EI_top50 = EI[top_50_indices]
    x_GD = args_history[x_random]
    
    best_point, max_val = x_GD[0], EI_top50[0]
    for x in x_GD:
        result = minimize(
            fun=lambda x: -ei_maximization(x.reshape(1, -1), GP, max_val),
            x0=x,
            bounds=x_random,
            options={'maxiter': 100}
        )
        if max_val < -result.fun:
            best_point, max_val = result.x, -result.fun
    
    return best_point