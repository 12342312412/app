import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from kde_ebm import mixture_model
from kde_ebm import mcmc
from kde_ebm import plotting
from kde_ebm import datasets

def main():
    st.title("KDE EBM Example with Streamlit")

    # Load data
    try:
        X, y, bmname, cname = datasets.load_synthetic('synthetic_1500_10.csv')
        st.write("Data loaded successfully.")
    except Exception as e:
        st.write(f"Error loading data: {e}")
        st.stop()

    # Display data
    st.subheader("CSV Data")
    st.write(pd.DataFrame(X, columns=bmname))

    # Fit GMM for each biomarker and plot the results
    try:
        mixture_models = mixture_model.fit_all_kde_models(X, y)
        fig, ax = plotting.mixture_model_grid(X, y, mixture_models, score_names=bmname, class_names=cname)
        st.pyplot(fig)
    except Exception as e:
        st.write(f"Error fitting GMM: {e}")
        st.stop()

    # Now we fit our disease sequence, using greedy ascent followed by MCMC optimisation
    try:
        res = mcmc.mcmc(X, mixture_models, n_iter=500, greedy_n_iter=10, greedy_n_init=5)
        fig, ax = plotting.mcmc_uncert_mat(res, score_names=bmname)
        st.pyplot(fig)
    except Exception as e:
        st.write(f"Error fitting disease sequence: {e}")
        st.stop()

    # Sort the results and get the maximum likelihood order
    res.sort(reverse=True)
    ml_order = res[0]

    # Stage all participants using the fitted EBM
    try:
        prob_mat = mixture_model.get_prob_mat(X, mixture_models)
        stages, stages_like = ml_order.stage_data(prob_mat)

        fig, ax = plt.subplots(figsize=(12, 6))  
        plotting.stage_histogram(stages, y, ax=ax)  
        st.pyplot(fig)

        # Display the stages as a DataFrame
        st.subheader("Participant Stages")
        stages_df = pd.DataFrame(stages, columns=["Stage"])
        stages_df["Class"] = y
        st.write(stages_df)
    except Exception as e:
        st.write(f"Error staging participants: {e}")
        st.stop()

if __name__ == '__main__':
    np.random.seed(42)
    main()
