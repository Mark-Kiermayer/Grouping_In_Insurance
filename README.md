# Grouping-in-insurance-with-neural-networks
Research project 2019-2020
Accompanying code to paper "Grouping of contracts in insurance using neural networks" by M. Kiermayer and C. Weiß. Available at https://www.tandfonline.com/doi/abs/10.1080/03461238.2020.1836676 .

# Goal:
Create a surrogate portfolio (-> grouping) that is (in some sense) simpler than an existing portfolio, but displayed characteristics that are as close as possible the the original portfolio. <br/>
The main motivation for this objective is the Solvency II directive and computationally highly expensive simulations. Simplifying the portfolio leads to time savings and is in practice essential. <br/>
A naive approach for such a grouping is the unsupervised K-means clustering algorithm and its cluster assignment. We improve this baseline by including an economic supervision (by neural networks). Further, our approach can also be used for fuzzy clustering.

# Methodology
This grouping procedure is based on a 2-step approach where
    i) a prediction model for characteristics of contracts is fitted
    ii) the grouping is performed, given a fixed prediction model that supervises the resulting aggregated characteristics

# Content of the github-project:

1) Generation of (realistic) data <br/>
    * Term life insurance contracts 
          + for training a prediction model (-> "SUB_DATA_TL.py")
          + for performing a grouping procedure (-> "SUB_DATA_TL_NEW_Portfolio.py")
    * Pension contracts, i.e. defined benefit plans 
          + for training a prediction model (-> "SUB_DATA_Pensions.py")
          + for performing a grouping procedure (-> "SUB_DATA_Pensions_NEW_Portfolio.py")

2) Calibration and analysis of the prediction models
    * For term life insurance: MAIN_Prediction_TL.py
    * For pension contracts:  MAIN_Prediction_Pensions.py
    
3) Grouping procedure
    * For term life portfolio: MAIN_Grouping_TL{}.py
    * For pension portfolio:  MAIN_Grouping_Pensions.py
    
4) Sensitivity analysis of the investment surplus for the term life portfolio (-> MAIN_Sensis_TL_SCR.py)


Note on 1): See references for realistic distribution of the data in the paper.
