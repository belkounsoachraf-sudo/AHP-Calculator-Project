import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Fonction de Calcul AHP Détaillée (Remplace ahp_core) ---
# Valeurs d'Indice de Cohérence Aléatoire (Random Consistency Index - RI)
RI_VALUES = {
    1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24,
    7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49
}

def calculate_ahp(matrix):
    n = matrix.shape[0]
    
    # Récupérer la valeur RI
    RI = RI_VALUES.get(n, 1.49) # Utilise 1.49 par défaut si n > 10

    # 1. Somme des Colonnes
    col_sums = np.sum(matrix, axis=0)
    
    # 2. Matrice Normalisée : chaque élément divisé par la somme de sa colonne
    # [np.newaxis, :] assure que col_sums est un vecteur ligne pour la division
    normalized_matrix = matrix / col_sums[np.newaxis, :]

    # 3. Vecteur de Priorité (Poids) - Moyenne des lignes de la matrice normalisée
    weights = np.mean(normalized_matrix, axis=1)

    # 4. Lambda Max (Valeur Propre Maximale)
    # Calculé comme la moyenne du vecteur (A * w) / w
    Aw = np.dot(matrix, weights)
    lambda_max = np.mean(Aw / weights)

    # 5. Indice de Cohérence (CI)
    CI = (lambda_max - n) / (n - 1) if n > 1 else 0.0

    # 6. Taux de Cohérence (CR)
    CR = CI / RI if RI > 0 else 0.0

    # 7. Message
    if CR <= 0.10:
        cr_message = "Les jugements sont suffisamment cohérents (CR ≤ 0.10). Les priorités sont fiables."
    else:
        cr_message = "Les jugements sont incohérents (CR > 0.10). Veuillez revoir vos comparaisons pour améliorer la fiabilité des priorités."

    # Retourne tous les résultats intermédiaires
    return col_sums, normalized_matrix, weights, lambda_max, CI, CR, RI, cr_message

# --- Configuration et Titre (Modification pour signature) ---
st.set_page_config(layout="wide")
st.title("🧮 Calculatrice AHP (Analytic Hierarchy Process)")
st.caption("Application interne pour l'aide à la décision multicritère | **Développeur : Achraf BELKOUNSO**")

# --- Étape 1 : Saisie des Éléments (Critères ou Alternatives) ---
st.header("1. Définition des Éléments")

element_list_str = st.text_area(
    "Liste des Éléments à Comparer (un par ligne, ex: Critère A, Critère B, ...)",
    "Coût\nPerformance\nSécurité"
)

# Convertir la chaîne de caractères en une liste de noms
elements = [e.strip() for e in element_list_str.split('\n') if e.strip()]
n = len(elements)

if n < 2:
    st.warning("Veuillez saisir au moins deux éléments pour la comparaison.")
else:
    st.success(f"Nombre d'éléments détectés : **{n}**")
    
    # --- Étape 2 : Saisie des Jugements (Matrice) ---
    st.header("2. Saisie de la Matrice de Comparaison par Paires (Échelle 1-9)")
    st.info("Saisissez seulement les valeurs au-dessus de la diagonale. Les valeurs inverses sont calculées automatiquement.")

    # Initialisation de la matrice de comparaison
    matrix = np.ones((n, n), dtype=float)
    
    # Création d'une interface de tableau pour la saisie
    with st.form("ahp_input_form"):
        # Affichage des labels des colonnes pour la saisie (Meilleure lisibilité)
        input_cols = st.columns(n)
        for k in range(n):
             with input_cols[k]:
                 if k > 0: # Cacher la première colonne de la zone de saisie
                     st.markdown(f"**{elements[k]}**")

        # Boucle pour la saisie interactive des inputs (seulement i < j)
        # Utilisation d'un format visuel matriciel plus clair
        for i in range(n):
            row_cols = st.columns(n)
            with row_cols[i]:
                 st.markdown(f"**{elements[i]}**")
            for j in range(i + 1, n):
                # La comparaison C_i vs C_j
                with row_cols[j]:
                    # Utiliser une clé unique pour chaque widget Streamlit
                    value = st.number_input(
                        f"Comparaison de {elements[i]} par rapport à {elements[j]}", 
                        min_value=1.0/9.0, max_value=9.0, value=1.0, 
                        step=0.01, format="%.2f", 
                        key=f"input_{i}_{j}",
                        label_visibility="collapsed" # Cacher le label long pour la matrice
                    )
                    # Mise à jour de la matrice
                    matrix[i, j] = value
                    matrix[j, i] = 1.0 / value  # Réciproque
            
            # Afficher un champ désactivé ou laisser vide pour les éléments i=j ou j<i non saisis
            for j in range(i + 1):
                 if i != j:
                    with row_cols[j]:
                        st.text_input(f"Input {i}{j}", value=f"1/{matrix[i,j]:.2f}", disabled=True, label_visibility="collapsed")
                 else:
                    with row_cols[j]:
                        st.text_input(f"Input {i}{j}", value="1.00", disabled=True, label_visibility="collapsed")

        st.markdown("---")
        submitted = st.form_submit_button("Calculer les Poids et la Cohérence")

    # --- Étape 3 : Affichage des Résultats ---
    if submitted:
        
        # Affichage de la Matrice construite
        df_matrix = pd.DataFrame(matrix, index=elements, columns=elements)
        st.subheader("3. Matrice de Comparaison Complète")
        st.dataframe(df_matrix.style.format("{:.3f}"))

        # Appel à la fonction de calcul AHP
        col_sums, normalized_matrix, weights, lambda_max, CI, CR, RI, message = calculate_ahp(matrix)

        st.header("4. Étapes Détaillées du Calcul AHP")
        
        # 4.1 Normalisation
        st.subheader("4.1 Normalisation de la Matrice")
        st.markdown("##### Somme des Colonnes de la Matrice de Comparaison")
        df_col_sums = pd.DataFrame([col_sums], columns=elements, index=['Somme des colonnes'])
        st.dataframe(df_col_sums.style.format("{:.3f}"))
        
        st.markdown("""
        Chaque élément est divisé par la somme de sa colonne.
        """)
        
        st.markdown("##### Matrice Normalisée")
        df_normalized_matrix = pd.DataFrame(normalized_matrix, index=elements, columns=elements)
        st.dataframe(df_normalized_matrix.style.format("{:.4f}"))

        # 4.2 Poids (Vecteur de Priorité)
        st.subheader("4.2 Calcul du Vecteur de Priorité (Poids)")
        
        st.markdown(f"""
        Le poids de chaque élément est la **moyenne des valeurs de sa ligne** dans la Matrice Normalisée.
        """)
        
        # Afficher la moyenne des lignes
        df_weights_step = pd.DataFrame(normalized_matrix, index=elements, columns=elements)
        df_weights_step['Poids (Moyenne)'] = weights
        st.dataframe(df_weights_step.style.format("{:.4f}"))


        # 4.3 Cohérence
        st.subheader("4.3 Calcul de la Cohérence")
        
        # Calcul de λ_max
        st.markdown("##### 1. Valeur Propre Maximale (λ_max)")
        st.markdown(f"La valeur propre maximale **($\lambda_{{\\text{{max}}}}$)** est : **{lambda_max:.4f}** (La valeur idéale pour une matrice parfaitement cohérente est $n={n}$).")
        
        # Calcul de CI
        st.markdown("##### 2. Indice de Cohérence (CI)")
        st.markdown(f"$$CI = \\frac{{\lambda_{{\\text{{max}}}} - n}}{{n - 1}} = \\frac{{{lambda_max:.4f} - {n}}}{{{n} - 1}} = \\text{{{CI:.4f}}}$$")
        
        # Calcul de CR
        st.markdown("##### 3. Taux de Cohérence (CR)")
        st.markdown(f"L'Indice Aléatoire (RI) pour $n={n}$ est **{RI:.4f}**.")
        st.markdown(f"$$CR = \\frac{{CI}}{{RI}} = \\frac{{{CI:.4f}}}{{{RI:.4f}}} = \\text{{{CR:.4f}}}$$")


        # --- Affichage Final des Poids et Cohérence ---
        st.header("5. Synthèse des Résultats")
        
        # 5.1 Cohérence Finale
        st.subheader("Taux de Cohérence Final")
        if CR <= 0.10:
            st.success(f"**Taux de Cohérence (CR) :** {CR:.4f} (Cohérent)")
        else:
            st.error(f"**Taux de Cohérence (CR) :** {CR:.4f} (Incohérent)")
            
        st.markdown(f"**Interprétation :** {message}")

        # 5.2 Priorités Finales
        st.subheader("Priorités (Poids) des Éléments")
        
        # Créer un DataFrame pour les résultats
        df_results = pd.DataFrame({
            'Élément': elements,
            'Poids (Priorité)': weights.round(4)
        }).sort_values(by='Poids (Priorité)', ascending=False).reset_index(drop=True)
        
        df_results['Poids (%)'] = (df_results['Poids (Priorité)'] * 100).round(2).astype(str) + ' %'
        
        st.dataframe(df_results, hide_index=True)
        
        # 5.3 Visualisation Graphique
        st.subheader("Visualisation Graphique des Priorités")
        
        fig, ax = plt.subplots()
        # Utiliser une palette de couleurs dynamique
        colors = plt.cm.get_cmap('Spectral', len(elements))
        ax.bar(df_results['Élément'], df_results['Poids (Priorité)'], color=colors(np.arange(len(elements))))
        ax.set_ylabel('Priorité / Poids')
        ax.set_title('Distribution des Poids AHP')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
        
        # --- 5.4 Détermination du Meilleur Choix (Nouveau) ---
        st.subheader("5.4 Recommandation de Priorité")
        
        # Le meilleur choix est le premier élément dans le DataFrame trié
        best_choice = df_results.iloc[0]['Élément']
        best_score_percent = df_results.iloc[0]['Poids (%)']

        st.success(f"Selon les pondérations AHP, l'élément le plus prioritaire est : **{best_choice}** avec un poids de **{best_score_percent}**.")
        
        if CR > 0.10:
             st.warning("Attention : Bien que cet élément soit le plus prioritaire, le Taux de Cohérence (CR) est élevé. Veuillez revoir vos jugements pour assurer la fiabilité de cette recommandation.")
