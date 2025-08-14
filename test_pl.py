import pytest
import numpy as np
from pl import solve_linear_programming, load_problem_from_json

def test_solve_linear_programming():
    """
    Testa a função solve_linear_programming com os dados do problema padrão.
    """
    # Carrega o problema do arquivo JSON
    problem = load_problem_from_json('problem.json')

    # Configura o problema
    c = np.array(problem['objective_coeffs'])
    A_ub = np.array(problem['inequality_constraint_matrix'])
    b_ub = np.array(problem['inequality_constraint_bounds'])
    A_eq = np.array(problem['equality_constraint_matrix']) if problem['equality_constraint_matrix'] is not None else None
    b_eq = np.array(problem['equality_constraint_bounds']) if problem['equality_constraint_bounds'] is not None else None
    bounds = problem['decision_variable_bounds']

    # Resolve o problema de LP
    result = solve_linear_programming(c, A_ub, b_ub, A_eq, b_eq, bounds)

    # Verifica se a otimização foi bem-sucedida
    assert result.success

    # Verifica o valor ótimo da função objetivo (com uma tolerância)
    # Valor esperado: -266.666...
    expected_fun = -266.666666
    assert abs(result.fun - expected_fun) < 1e-4

    # Verifica os valores ótimos das variáveis de decisão (com uma tolerância)
    # Valores esperados: [0, 33.333...]
    expected_x = np.array([0, 33.333333])
    np.testing.assert_allclose(result.x, expected_x, atol=1e-4)
