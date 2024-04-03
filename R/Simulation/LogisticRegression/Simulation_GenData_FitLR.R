# This R script:
  # samples data sets from the parent simulation data set
  # saves the data sets
  # fits a logistic regression on each of the data sets
  # saves the results from the logistic regression for analysis

# ----*** RE-RUN FOR EACH SAMPLE SIZE (n)***----

# load packages
library(mosaic)
library(dplyr)
library(tidyverse)
library(ggplot2)
library(RColorBrewer)
library(knitr)
library(ggplot2)
library(GGally)
library(MASS)
library(reshape2)
library(pROC)

# read in the data

compas_sim_path <-
  "/home/dasienga24/Statistics-Senior-Honors-Thesis/Data Sets/COMPAS/compas_sim.csv"
compas_sim_parent <- read.csv(compas_sim_path)

# set useful values

samplesizes <- c(500, 1000, 2500, 5000) #desired sample sizes
set.seed(123) # for reproducibility

for (n in samplesizes) {
  
  reps <- 250 # number of repetitions for each sample size
  
  # initialize storage vectors
  
  lr_convergence <- rep(0, reps)
  lr_accuracy <- rep(0, reps)
  lr_discrimination <- rep(0, reps)
  dataset_id <- rep(0, reps)
  
  # run the simulation
  
  for (i in 1:reps) {
    # sample data set from the parent data set
    df <- sample(compas_sim_parent, n, replace = TRUE) |>
      dplyr::select(c(race, age, prior_offense, is_recid))
    
    # save the data set for later use with the Seldonian simulation
    write.csv(
      df,
      file = paste0(
        "/home/dasienga24/Statistics-Senior-Honors-Thesis/Data Sets/SimulationData/sim_",
        n,
        "_",
        i,
        ".csv"
      )
    )
    
    # fit the logistic regression
    lr <- glm(is_recid ~ age + prior_offense,
              data = df,
              family = binomial(logit))
    
    # calculate the discrimination
    preds <- predict(lr, newdata = df, type = "response")
    
    df <- df %>%
      mutate(
        preds = preds,
        prediction = round(preds, 0),
        pred_risk = ifelse(prediction == 0, 'Low', 'High')
      )
    
    discrimination <- df %>%
      dplyr::select(race, pred_risk, is_recid) %>%
      group_by(race, is_recid) %>%
      mutate(total = n()) %>%
      group_by(pred_risk, race, total) %>%
      summarise(
        "reoffended" = count(is_recid == 1),
        "did_not_reoffend" = count(is_recid == 0)
      ) %>%
      pivot_longer(cols = c("reoffended", "did_not_reoffend"),
                   names_to = "recidivism") %>%
      pivot_wider(
        id_cols = c("pred_risk", "recidivism", "total"),
        names_from = "race",
        values_from = value
      ) %>%
      rename("Black" = `African-American`,
             "White" = `Caucasian`) %>%
      mutate(Black = round(100 * Black / total, 2),
             White = round(100 * White / total, 2)) %>%
      dplyr::select(-total) %>%
      group_by(pred_risk, recidivism) %>%
      summarize(Black = max(Black, na.rm = TRUE),
                White = max(White, na.rm = TRUE)) %>%
      filter((pred_risk == "High" &
                recidivism == "did_not_reoffend") |
               (pred_risk == "Low" & recidivism == "reoffended")
      )
    
    # save metrics
    dataset_id[i] <- i
    lr_convergence[i] <- lr[["converged"]]
    lr_accuracy[i] <-
      count(round(lr[["fitted.values"]]) == lr[["y"]]) / nrow(df)
    lr_discrimination[i] <-
      sum(abs(discrimination$White - discrimination$Black)) / 100
    
  }
  
  # synthesize results
  
  lr_output <-
    cbind(dataset_id, lr_convergence, lr_accuracy, lr_discrimination) |>
    as.data.frame()
  
  # save results
  
  write.csv(
    lr_output,
    file = paste0(
      "/home/dasienga24/Statistics-Senior-Honors-Thesis/R/Simulation/LogisticRegression/Results/lr_",
      n,
      ".csv"
    )
  )
}
