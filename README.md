# Indian-American Campaign Contributions, 1998-2022
https://indianamericancontributions.vercel.app/

This code generates the datasets and figures used for our paper "An Emerging Lobby: An Analysis of Campaign Contributions from Indian-Americans, 1998-2022" by Karnav Popat, Vishnu Prakash & Joyojeet Pal. All data is from the OpenSecrets campaign finance releases.

The four main notebooks are:

 - donor_stats (all the data and figures relating to the Indian-American population and raw numbers of donors and contributions)
 - donor_maps (the map figures relating to donors and contributions)
 - donor_recip_stats (the data and figures relating to the party leaning of Indian-American and other contributors)
 - donor_recip_maps (the map figures relating to party leaning)

The code used for the name-based ethnicity classification is in:

 - classif/name_stats_prediction_yearly (the first step, where names are classified based on the naive popularity in India versus the United States)
 - classif/classif_ethnicia (the second step, which trains the n-gram model to classify the dataset)

Note that this is unfortunately not the final version of the code for the model used in our paper, as we modified the way non-exclusively-Indian names are handled, and undertook a third step of manual verification and annotation for high-uncertainty names and high-contributing donors.

To replicate our analysis, follow these steps:

 - Download the Campaign Finance files from OpenSecrets for the electoral cycles 2000 to 2022
 - Run data_agg.py to aggregate the data from transaction-level to donor-per-cycle level
 - Run the classif/ code and manually verify/annotate to generate your ethnicity annotations
 - Run data_pred_agg.py to add these predictions to the dataset
 - Run data_pred_recip_agg.py to generate the dataset at the level of donor-and-recipient-per-cycle
 - Run the four notebooks with the appropriate year and file location to generate the figures in images/
