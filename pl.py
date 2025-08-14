import numpy as np
from scipy.optimize import linprog

# --- Configurações e Parâmetros do Problema ---

# Coeficientes da função objetivo (para minimização)
# Queremos maximizar Z = 5*xA + 8*xB
# Para minimizar, usamos -Z = -5*xA - 8*xB
OBJECTIVE_COEFFS = np.array([-5, -8])

# Coeficientes das restrições de desigualdade (Ax <= b)
# Restrição 1: 2*xA + 3*xB <= 100
# Restrição 2: 4*xA + 2*xB <= 80
INEQUALITY_CONSTRAINT_MATRIX = np.array([
    [2, 3],  # Coeficientes para xA e xB na primeira restrição
    [4, 2]   # Coeficientes para xA e xB na segunda restrição
])

# Limites do lado direito das restrições de desigualdade
INEQUALITY_CONSTRAINT_BOUNDS = np.array([100, 80])

# Coeficientes das restrições de igualdade (Ax = b) - Não temos nenhuma neste exemplo
EQUALITY_CONSTRAINT_MATRIX = None
EQUALITY_CONSTRAINT_BOUNDS = None

# Limites para as variáveis de decisão (xA >= 0, xB >= 0)
# Cada tupla é (limite_inferior, limite_superior) para cada variável.
# None significa que não há limite (infinito).
DECISION_VARIABLE_BOUNDS = [(0, None), (0, None)]

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

def display_lp_results(result, variable_names=["Produção Produto A", "Produção Produto B"]):
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

def main():
    """
    Define o problema de LP e executa a solução e exibição dos resultados.
    """
    # Configura o problema
    c = OBJECTIVE_COEFFS
    A_ub = INEQUALITY_CONSTRAINT_MATRIX
    b_ub = INEQUALITY_CONSTRAINT_BOUNDS
    A_eq = EQUALITY_CONSTRAINT_MATRIX
    b_eq = EQUALITY_CONSTRAINT_BOUNDS
    bounds = DECISION_VARIABLE_BOUNDS

    # Resolve o problema de LP
    lp_result = solve_linear_programming(c, A_ub, b_ub, A_eq, b_eq, bounds)

    # Exibe os resultados
    display_lp_results(lp_result)

# --- Ponto de Entrada do Script ---

if __name__ == "__main__":
    # Este bloco garante que main() só será executado quando o script for rodado
    # diretamente (não quando importado como módulo em outro script).
    main()
