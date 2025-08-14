import numpy as np
from scipy.optimize import linprog
import json
import argparse

# --- Funções ---

def solve_linear_programming(c, A_ub, b_ub, A_eq, b_eq, bounds):
    """
    Resolve um problema de Programação Linear usando scipy.optimize.linprog.

    Args:
        c (np.ndarray): Coeficientes da função objetivo a ser minimizada.
        A_ub (np.ndarray): Matriz de coeficientes para as restrições de desigualdade (Ax <= b).
        b_ub (np.ndarray): Vetor de limites do lado direito para as restrições de desigualdade.
        A_eq (np.ndarray or None): Matriz de coeficientes para as restrições de igualdade (Ax = b).
        b_eq (np.ndarray or None): Vetor de limites do lado direito para as restrições de igualdade.
        bounds (list of tuple): Lista de tuplas definindo os limites (inferior, superior)
                                 para cada variável de decisão.

    Returns:
        OptimizeResult: Um objeto contendo os resultados da otimização.
    """
    print("Iniciando a solução do problema de Programação Linear...")
    # Usamos o método 'highs', que é geralmente o mais robusto e recomendado
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    print("Solução concluída.")
    return result

def display_lp_results(result, variable_names):
    """
    Exibe os resultados de um problema de Programação Linear de forma legível.

    Args:
        result (OptimizeResult): O objeto de resultado retornado por linprog.
        variable_names (list of str): Nomes descritivos para as variáveis de decisão.
    """
    if result.success:
        print("\n--- Resultados da Otimização ---")
        # O valor de result.fun é o valor mínimo da função objetivo minimizada.
        # Como maximizamos -Z, o valor máximo de Z é -result.fun.
        print(f"Valor Ótimo da Função Objetivo (Lucro Máximo): {(-result.fun):.2f}")
        print("\nVariáveis de Decisão Ótimas:")
        for i, name in enumerate(variable_names):
            print(f"  {name}: {result.x[i]:.2f}")
        print("------------------------------")
    else:
        print("\n--- Erro na Otimização ---")
        print(f"O problema de Programação Linear não pôde ser resolvido.")
        print(f"Status: {result.message}")
        print("--------------------------")

# --- Lógica Principal de Execução ---

def load_problem_from_json(file_path):
    """
    Carrega a definição do problema de um arquivo JSON.
    """
    with open(file_path, 'r') as f:
        problem = json.load(f)

    # Converte listas para tuplas nos limites das variáveis
    problem['decision_variable_bounds'] = [
        tuple(bounds) for bounds in problem['decision_variable_bounds']
    ]
    return problem

def main():
    """
    Define o problema de LP e executa a solução e exibição dos resultados.
    """
    # Configura o parser de argumentos
    parser = argparse.ArgumentParser(description='Resolve um problema de Programação Linear a partir de um arquivo JSON.')
    parser.add_argument(
        '-f', '--file',
        default='problem.json',
        help='Caminho para o arquivo JSON com a definição do problema.'
    )
    args = parser.parse_args()

    # Carrega o problema do arquivo JSON
    try:
        problem = load_problem_from_json(args.file)
    except FileNotFoundError:
        print(f"Erro: O arquivo '{args.file}' não foi encontrado.")
        return
    except json.JSONDecodeError:
        print(f"Erro: O arquivo '{args.file}' não é um JSON válido.")
        return

    # Configura o problema
    c = np.array(problem['objective_coeffs'])
    A_ub = np.array(problem['inequality_constraint_matrix'])
    b_ub = np.array(problem['inequality_constraint_bounds'])
    A_eq = np.array(problem['equality_constraint_matrix']) if problem['equality_constraint_matrix'] is not None else None
    b_eq = np.array(problem['equality_constraint_bounds']) if problem['equality_constraint_bounds'] is not None else None
    bounds = problem['decision_variable_bounds']
    variable_names = problem['variable_names']

    # Resolve o problema de LP
    lp_result = solve_linear_programming(c, A_ub, b_ub, A_eq, b_eq, bounds)

    # Exibe os resultados
    display_lp_results(lp_result, variable_names)

# --- Ponto de Entrada do Script ---

if __name__ == "__main__":
    # Este bloco garante que main() só será executado quando o script for rodado
    # diretamente (não quando importado como módulo em outro script).
    main()
