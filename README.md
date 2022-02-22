# Grouping-in-insurance-with-neural-networks
Research project 2019-2020.<br/>
Accompanying code to paper "Grouping of contracts in insurance using neural networks" by M. Kiermayer and C. Weiß.<br/> Paper available at https://www.tandfonline.com/doi/abs/10.1080/03461238.2020.1836676 .

# Goal:
Create a surrogate portfolio (-> grouping) that is (in some sense) simpler than an existing portfolio, but displayed characteristics that are as close as possible the the original portfolio. <br/>
The main motivation for this objective is the Solvency II directive and computationally highly expensive simulations. Simplifying the portfolio leads to time savings and is in practice essential. <br/>
A naive approach for such a grouping is the unsupervised K-means clustering algorithm and its cluster assignment. We improve this baseline by including an economic supervision (by neural networks). Further, our approach can also be used for fuzzy clustering.

# Methodology
This grouping procedure is based on a 2-step approach where<br/>
    i) a prediction model for characteristics of contracts is fitted<br/>
    ii) the grouping is performed, given a fixed prediction model that supervises the resulting aggregated characteristics <br/>

# Content of the github-project:

1) Generation of (realistic) data <br/>
    * Term life insurance contracts <br/>
          + for training a prediction model (-> "SUB_DATA_TL.py")<br/>
          + for performing a grouping procedure (-> "SUB_DATA_TL_NEW_Portfolio.py")<br/>
    * Pension contracts, i.e. defined benefit plans <br/>
          + for training a prediction model (-> "SUB_DATA_Pensions.py")<br/>
          + for performing a grouping procedure (-> "SUB_DATA_Pensions_NEW_Portfolio.py")<br/>

2) Calibration and analysis of the prediction models<br/>
    * For term life insurance: MAIN_Prediction_TL.py<br/>
    * For pension contracts:  MAIN_Prediction_Pensions.py<br/>
    
3) Grouping procedure<br/>
    * For term life portfolio: MAIN_Grouping_TL{}.py<br/>
    * For pension portfolio:  MAIN_Grouping_Pensions.py<br/>
    
4) Sensitivity analysis of the investment surplus for the term life portfolio (-> MAIN_Sensis_TL_SCR.py)<br/>


Note on 1): See references for realistic distribution of the data in the paper.
