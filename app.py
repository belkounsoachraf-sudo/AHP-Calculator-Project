import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Fonction de Calcul AHP Détaillée (Reprise de la version précédente) ---
RI_VALUES = {
    1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24,
    7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49
}

def calculate_ahp(matrix):
    n = matrix.shape[0]
    if n == 0:
        return None, None, None, None, None, None, None, "Matrice vide."
        
    RI = RI_VALUES.get(n, 1.49)
    col_sums = np.sum(matrix, axis=0)
    normalized_matrix = matrix / col_sums[np.newaxis, :]
    weights = np.mean(normalized_matrix, axis=1)

    Aw = np.dot(matrix, weights)
    lambda_max = np.mean(Aw / weights)

    CI = (lambda_max - n) / (n - 1) if n > 1 else 0.0
    CR = CI / RI if RI > 0 else 0.0

    if CR <= 0.10:
        cr_message = "Cohérent (CR ≤ 0.10). Priorités fiables."
    else:
        cr_message = "Incohérent (CR > 0.10). Revoir les jugements."

    return col_sums, normalized_matrix, weights, lambda_max, CI, CR, RI, cr_message

# --- Configuration de l'Application ---
st.set_page_config(layout="wide")
st.title("🥇 AHP Multi-Niveaux : Choix d'un Candidat")
st.caption("Application interne pour la pondération des critères et l'évaluation des alternatives | Développeur : Achraf BELKOUNSO")

# --- Initialisation de l'état de session (pour stocker les données) ---
if 'criteria' not in st.session_state:
    st.session_state['criteria'] = ["Expérience", "Compétences", "Adaptabilité", "Coût Salarial"]
if 'candidates' not in st.session_state:
    st.session_state['candidates'] = ["Candidat A", "Candidat B", "Candidat C"]
if 'n_crit' not in st.session_state:
    st.session_state['n_crit'] = 4
if 'n_cand' not in st.session_state:
    st.session_state['n_cand'] = 3
if 'W_crit' not in st.session_state:
    st.session_state['W_crit'] = None
if 'W_cand_by_crit' not in st.session_state:
    st.session_state['W_cand_by_crit'] = {}

# --- Fonction pour générer l'interface de saisie de matrice ---
def input_matrix_form(elements, matrix_key, title):
    n = len(elements)
    
    # Récupérer la matrice existante ou initialiser
    if matrix_key not in st.session_state:
        st.session_state[matrix_key] = np.ones((n, n), dtype=float)
    matrix = st.session_state[matrix_key]
    
    st.subheader(title)
    st.info("Saisissez les jugements d'importance relative (Échelle 1-9).")

    with st.form(f"form_{matrix_key}"):
        
        # Affichage des labels des colonnes
        cols_labels = st.columns(n)
        for k in range(n):
            with cols_labels[k]:
                st.markdown(f"**{elements[k]}**")

        # Saisie des inputs
        for i in range(n):
            row_cols = st.columns(n)
            with row_cols[i]:
                st.markdown(f"**{elements[i]}**")
            
            for j in range(i + 1, n):
                with row_cols[j]:
                    # Utiliser une clé unique pour chaque widget Streamlit
                    value = st.number_input(
                        f"Comparaison {elements[i]} vs {elements[j]}", 
                        min_value=1.0/9.0, max_value=9.0, value=matrix[i, j], 
                        step=0.01, format="%.2f", 
                        key=f"input_{matrix_key}_{i}_{j}",
                        label_visibility="collapsed"
                    )
                    matrix[i, j] = value
                    matrix[j, i] = 1.0 / value  # Réciproque

            # Afficher les valeurs réciproques (pour la lisibilité)
            for j in range(i + 1):
                with row_cols[j]:
                    val_str = "1.00" if i == j else f"1/{matrix[i, j]:.2f}"
                    st.text_input(f"Display_{matrix_key}_{i}_{j}", value=val_str, disabled=True, label_visibility="collapsed")


        submitted = st.form_submit_button("Calculer la Priorité (Poids)")
        
        if submitted:
            # Stocker la matrice mise à jour
            st.session_state[matrix_key] = matrix
            
            # Calcul AHP
            col_sums, norm_matrix, weights, lambda_max, CI, CR, RI, message = calculate_ahp(matrix)
            
            # Affichage des résultats
            if weights is not None:
                st.subheader("Résultats du Calcul AHP")
                
                # Affichage de la Cohérence
                if CR <= 0.10:
                    st.success(f"CR : {CR:.4f} ({message})")
                else:
                    st.error(f"CR : {CR:.4f} ({message})")

                # Affichage des Poids
                df_weights = pd.DataFrame({'Élément': elements, 'Poids': weights.round(4)})
                st.dataframe(df_weights, hide_index=True)
                
                return weights
    return None

# --- Interface Utilisateur Principale ---

tab1, tab2, tab3 = st.tabs(["1. Configuration & Critères", "2. Évaluation des Candidats", "3. Synthèse Finale"])

# ====================================================================
# --- TAB 1: Configuration et Critères ---
# ====================================================================
with tab1:
    st.header("1.1 Configuration des Éléments")
    
    col_crit, col_cand = st.columns(2)
    
    with col_crit:
        st.subheader("Critères de Choix")
        criteria_str = st.text_area(
            "Liste des Critères (un par ligne)",
            "\n".join(st.session_state['criteria'])
        )
        st.session_state['criteria'] = [e.strip() for e in criteria_str.split('\n') if e.strip()]
        st.session_state['n_crit'] = len(st.session_state['criteria'])
        st.info(f"Nombre de Critères : **{st.session_state['n_crit']}**")
        
    with col_cand:
        st.subheader("Alternatives (Candidats)")
        candidates_str = st.text_area(
            "Liste des Candidats (un par ligne)",
            "\n".join(st.session_state['candidates'])
        )
        st.session_state['candidates'] = [e.strip() for e in candidates_str.split('\n') if e.strip()]
        st.session_state['n_cand'] = len(st.session_state['candidates'])
        st.info(f"Nombre de Candidats : **{st.session_state['n_cand']}**")

    st.markdown("---")
    
    if st.session_state['n_crit'] > 1:
        st.header("1.2 Pondération des Critères")
        
        # Appel à la fonction pour la matrice des critères
        weights_crit = input_matrix_form(
            st.session_state['criteria'], 
            'matrix_criteria', 
            "Comparaison des Critères entre Eux (Importance Globale)"
        )
        
        if weights_crit is not None:
            st.session_state['W_crit'] = weights_crit
            st.success("Pondération des Critères calculée et enregistrée ! Passez à l'Étape 2.")
            
# ====================================================================
# --- TAB 2: Évaluation des Candidats par Critère ---
# ====================================================================
with tab2:
    st.header("2. Évaluation des Candidats")
    
    if st.session_state['W_crit'] is None:
        st.warning("Veuillez d'abord compléter l'Étape 1 (Pondération des Critères).")
    elif st.session_state['n_cand'] < 2:
        st.warning("Veuillez saisir au moins deux candidats à évaluer dans l'Étape 1.")
    else:
        st.success(f"Évaluation des **{st.session_state['n_cand']}** candidats selon **{st.session_state['n_crit']}** critères.")
        
        # Générer une interface de saisie de matrice pour CHAQUE critère
        for i, criterion in enumerate(st.session_state['criteria']):
            st.markdown(f"### ➡️ Évaluation pour le Critère : **{criterion}**")
            
            # Clé unique pour la matrice de ce critère
            matrix_key = f'matrix_cand_{i}'
            
            weights_cand = input_matrix_form(
                st.session_state['candidates'], 
                matrix_key, 
                f"Comparaison des Candidats selon le critère : {criterion}"
            )
            
            if weights_cand is not None:
                st.session_state['W_cand_by_crit'][criterion] = weights_cand
                st.success(f"Priorités des candidats enregistrées pour le critère **{criterion}**.")
            
        # Vérifier si toutes les matrices ont été calculées
        if len(st.session_state['W_cand_by_crit']) == st.session_state['n_crit']:
             st.markdown("---")
             st.balloons()
             st.success("Toutes les évaluations sont complètes ! Passez à l'Étape 3 pour la synthèse.")

# ====================================================================
# --- TAB 3: Synthèse Finale ---
# ====================================================================
with tab3:
    st.header("3. Synthèse et Décision Finale")
    
    if len(st.session_state['W_cand_by_crit']) != st.session_state['n_crit'] or st.session_state['W_crit'] is None:
        st.warning("Veuillez compléter toutes les matrices des Étapes 1 et 2 pour obtenir la synthèse finale.")
    else:
        # --- 3.1 Construction du Tableau Récapitulatif ---
        st.subheader("3.1 Tableau Récapitulatif AHP")
        
        # Crée un DataFrame avec les candidats comme colonnes
        df_recap = pd.DataFrame(index=st.session_state['criteria'])
        
        # Remplir avec les poids locaux des candidats pour chaque critère
        for cand in st.session_state['candidates']:
            df_recap[cand] = 0.0 # Initialisation
            
        for criterion in st.session_state['criteria']:
            weights = st.session_state['W_cand_by_crit'][criterion]
            
            # Assigner les poids locaux à la ligne du critère correspondant
            for i, cand in enumerate(st.session_state['candidates']):
                 df_recap.loc[criterion, cand] = weights[i]
        
        # Ajouter la colonne des Poids Globaux des Critères
        df_recap['Poids Global du Critère'] = st.session_state['W_crit']
        
        st.dataframe(df_recap.style.format("{:.4f}"))
        
        st.info("Chaque cellule Candidat/Critère contient la priorité locale du candidat pour ce critère.")

        # --- 3.2 Calcul du Score Final ---
        st.subheader("3.2 Calcul et Résultats Finaux")
        
        final_scores = {}
        for cand in st.session_state['candidates']:
            # Score Final = Somme (Poids Local * Poids Global)
            final_scores[cand] = np.sum(df_recap[cand].values * df_recap['Poids Global du Critère'].values)
            
        df_final_results = pd.DataFrame(
            list(final_scores.items()), 
            columns=['Candidat', 'Score Final AHP']
        ).sort_values(by='Score Final AHP', ascending=False).reset_index(drop=True)
        
        df_final_results['Score Final (%)'] = (df_final_results['Score Final AHP'] * 100).round(2).astype(str) + ' %'
        
        st.dataframe(df_final_results, hide_index=True)
        
        # --- 3.3 Recommandation ---
        st.markdown("---")
        st.subheader("3.3 Recommandation Finale")
        
        best_choice = df_final_results.iloc[0]['Candidat']
        best_score_percent = df_final_results.iloc[0]['Score Final (%)']
        
        st.success(f"🎉 Le **meilleur choix** selon l'Analyse AHP Multi-Niveaux est : **{best_choice}** avec un score final de **{best_score_percent}**.")
        
        # --- 3.4 Visualisation ---
        st.subheader("Visualisation Graphique des Scores")
        fig, ax = plt.subplots()
        colors = plt.cm.get_cmap('viridis', st.session_state['n_cand'])
        ax.bar(df_final_results['Candidat'], df_final_results['Score Final AHP'], color=colors(np.arange(st.session_state['n_cand'])))
        ax.set_ylabel('Score Final AHP')
        ax.set_title('Score Final Pondéré des Candidats')
        plt.xticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig)
