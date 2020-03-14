# Color ML

This repository contains code on data that was used to build models that can predict the color of MOFs using ML techniques.

## Notebooks

- `survey_eda.ipynb`: exploratory data analysis of the survey results. We run the survey using the [colorjeopardy flask app](https://github.com/kjappelbaum/colorjeopardy). Technical details about the survey can be found there. In the notebook we also perfom comparision to the xkcd results and drop entries with a particularly short or long response time. 
- `ml_baseline.ipynb`: contains some first model building steps.

## Data

- `backup_colorcrawler.csv`: snapshot of the colorjeopardy survey results
- `clean_survey.csv`: cleaned survey data in which we dropped results with short or long response times.
- `xkcd.tsv`: results from the xkcd survey which we downloaded on 24/2/2020 from https://xkcd.com/color/rgb.txt.
- `CoRE12K_testdata.csv` and `CoRE12K_traindata.csv`, train/test data as provided by S.M. Mosavi and used in his diversity study. Featurization explained there (columns include RACs, geometric features and also gas uptake properties).
- `annotated_df.csv` is the data extracted from the CSD which includes the color labels deposited there. 
- `cleaned_survey_unweighted_mean.csv` is the mean aggregated cleaned survey data (string - mean rgb mapping). 
- `cleaned_survey_unweighted_median.csv` is the median aggregated cleaned survey data (string - median rgb mapping).
- `color_feat_merged.csv` is the color data string from the CSD merged with the dataframe containing the descriptors.

Unweighted mean/median refers to the fact that we perform simple averaging in RGB space without taking into account that the percpetion of the human eye does not vary uniformly within this space. 

## Conventions

- For commit messages, we use [conventional commits](https://www.conventionalcommits.org/en/v1.0.0-beta.2/). `feat` are also commits that introduce a new analysis step.  
