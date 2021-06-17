# Predicting Item Preknowledge Using XGBoost trained with simulated data

A repository to play with an idea of simulating item preknowledge data using a modified version of van der Linden's Hierarchical IRT model. Below is the general outline of the procedure:

- Fit van der linden's Hierarchical IRT model to the dataset under investigation and estimate the item parameters for response time component and item response components.

- Simulate data using the same item parameters with a twist to embed item preknowledge for some hypothetical cheaters

- Train an XGBoost model using the simulated data

- Use the XGBoost model to predict cases in the real dataset under investigation. 

