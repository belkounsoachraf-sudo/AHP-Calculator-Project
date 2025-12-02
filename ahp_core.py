import numpy as np

# Tableau des Indices Aléatoires (RI) pour n=1 à n=10
RI = {
    1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
    6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49
}

def calculate_ahp(matrix_data):
    """
    Calcule le vecteur de priorité (poids) et le taux de cohérence (CR).
    :param matrix_data: Matrice numpy de comparaison par paires.
    :return: (weights, CR, consistency_message)
    """
    n = matrix_data.shape[0]

    # 1. Calcul des valeurs propres et vecteurs propres
    eigen_values, eigen_vectors = np.linalg.eig(matrix_data)
    
    # Trouver l'indice de la valeur propre maximale (lambda_max)
    lambda_max_index = np.argmax(eigen_values.real)
    lambda_max = eigen_values.real[lambda_max_index]
    
    # Extraire le vecteur propre correspondant et le normaliser
    weights = eigen_vectors.real[:, lambda_max_index]
    weights = weights / np.sum(weights)

    # 2. Calcul de l'Indice de Cohérence (CI)
    if n > 1:
        CI = (lambda_max - n) / (n - 1)
    else:
        CI = 0.0

    # 3. Calcul du Taux de Cohérence (CR)
    if n > 2:
        ri_value = RI.get(n, 1.49)  # Utilise 1.49 si n > 10 (rare en académie)
        CR = CI / ri_value
    else:
        CR = 0.0 # Toujours cohérent pour n=1 ou n=2

    # 4. Message de Cohérence
    if CR <= 0.10:
        message = "✅ Cohérence Acceptable (CR < 0.10)"
    else:
        message = "❌ Cohérence Inacceptable (CR > 0.10). Réviser les jugements."
        
    return weights, CR, message