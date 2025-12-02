import streamlit as st
import numpy as np
import pandas as pd
from ahp_core import calculate_ahp

# --- Configuration et Titre ---
st.set_page_config(layout="wide")
st.title("🧮 Calculatrice AHP (Analytic Hierarchy Process)")
st.caption("Application interne pour l'aide à la décision multicritère")

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
    df_input = pd.DataFrame(index=elements, columns=elements)

    with st.form("ahp_input_form"):
        cols = st.columns(n)
        
        # Boucle pour la saisie interactive des inputs (seulement i < j)
        for i in range(n):
            for j in range(i + 1, n):
                # La comparaison C_i vs C_j
                with cols[j]:
                    # Utilisez une clé unique pour chaque widget Streamlit
                    value = st.number_input(
                        f"{elements[i]} vs {elements[j]}", 
                        min_value=1.0/9.0, max_value=9.0, value=1.0, 
                        step=0.01, format="%.2f", 
                        key=f"input_{i}_{j}"
                    )
                    # Mise à jour de la matrice
                    matrix[i, j] = value
                    matrix[j, i] = 1.0 / value  # Réciproque

        submitted = st.form_submit_button("Calculer les Poids et la Cohérence")

    # --- Étape 3 : Affichage des Résultats ---
    if submitted:
        st.header("3. Résultats de l'Analyse AHP")
        
        # Affichage de la Matrice construite
        df_matrix = pd.DataFrame(matrix, index=elements, columns=elements)
        st.subheader("Matrice de Comparaison Complète")
        st.dataframe(df_matrix.style.format("{:.3f}"))

        # Appel à la fonction de calcul AHP
        weights, CR, message = calculate_ahp(matrix)

        # 3.1 Affichage de la Cohérence
        st.subheader("Taux de Cohérence")
        if CR <= 0.10:
            st.success(f"**Taux de Cohérence (CR) :** {CR:.4f}")
        else:
            st.error(f"**Taux de Cohérence (CR) :** {CR:.4f}")
            
        st.markdown(f"**Interprétation :** {message}")

        # 3.2 Affichage des Poids
        st.subheader("Priorités (Poids) des Éléments")
        
        # Créer un DataFrame pour les résultats
        df_results = pd.DataFrame({
            'Élément': elements,
            'Poids (Priorité)': weights.round(4)
        }).sort_values(by='Poids (Priorité)', ascending=False).reset_index(drop=True)
        
        df_results['Poids (%)'] = (df_results['Poids (Priorité)'] * 100).round(2).astype(str) + ' %'
        
        st.dataframe(df_results, hide_index=True)
        
        # 3.3 Visualisation Graphique
        st.subheader("Visualisation des Poids")
        
        # Utilisez Matplotlib pour un graphique simple (facile avec Streamlit)
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.bar(df_results['Élément'], df_results['Poids (Priorité)'], color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
        ax.set_ylabel('Priorité / Poids')
        ax.set_title('Distribution des Poids AHP')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)