# Script Setting and Resources
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
library(tidyverse)
library(haven) 
library(caret)
library(janitor)

# Data Import and Cleaning
gss_data <- read_spss("../data/GSS2016.sav")
gss_tbl <- gss_data %>%
  filter(!is.na(MOSTHRS)) %>%
  rename(`work hours` = MOSTHRS) %>%
  select(-HRS1, -HRS2) %>%
  remove_empty("cols", cutoff = 0.25) %>%  
  sapply(as.numeric) %>%
  as_tibble()

# Visualization
gss_tbl %>%
  ggplot(aes(x = `work hours`)) +
  geom_histogram()

# Analysis
set.seed(123) # For reproducibility
# Shuffle and split the dataset
gss_shuffled <- gss_tbl %>% sample_frac()
split_point <- round(nrow(gss_shuffled) * 0.75)

# Creating training and testing sets
gss_train <- gss_shuffled[1:split_point, ]
gss_test <- gss_shuffled[(split_point + 1):nrow(gss_shuffled), ]

# Proceed with the rest of the setup
fold_indices <- createFolds(gss_train_tbl$`work hours`, 
                            k = 10)

myControl <- trainControl(method = "cv", 
                          index = fold_indices, 
                          number = 10, 
                          verboseIter = TRUE)

# OLS Regression
model_ols <- train(`work hours` ~ ., 
                   data = gss_train, 
                   method = "lm", 
                   metric = "Rsquared", 
                   preProcess = "medianImpute", 
                   na.action = na.pass, 
                   trControl = myControl)
ols_predict <- predict(model_ols, gss_test)

# Elastic Net Model
model_elastic <- train(`work hours` ~ ., 
                       data = gss_train, 
                       method = "glmnet", 
                       preProcess = "medianImpute", 
                       na.action = na.pass, 
                       trControl = myControl)
elastic_predict <- predict(model_elastic, gss_test)

# Random Forest Model
model_rf <- train(`work hours` ~ ., 
                  data = gss_train, 
                  method = "ranger", 
                  preProcess = "medianImpute", 
                  na.action = na.pass, 
                  trControl = myControl)
rf_predict <- predict(model_rf, gss_test)

# eXtreme Gradient Boosting Model
model_xgb <- train(`work hours` ~ ., 
                   data = gss_train, 
                   method = "xgbLinear", 
                   preProcess = "medianImpute", 
                   na.action = na.pass, 
                   trControl = myControl)
xgb_predict <- predict(model_xgb, gss_test)


