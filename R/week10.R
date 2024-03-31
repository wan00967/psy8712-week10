# Script Setting and Resources
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
library(tidyverse)
library(haven) 
library(caret)
library(janitor)
library(glmnet)
library(ranger)

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
# Shuffle and split dataset
mod_vec = c("lm", "glmnet", "ranger", "xgbTree")
index = createDataPartition(gss_tbl$`work hours`, p = 0.75, list = FALSE)
gss_tbl_train = gss_tbl[index,]
gss_tbl_test = gss_tbl[-index,]

# 10 folds used in cross-validation from training set
training_folds = createFolds(gss_tbl_train$`work hours`, 10)
# reusable trainControl for all models 
reuseControl = trainControl( method = "cv", number = 10, search = "grid", 
                             indexOut = training_folds, verboseIter = TRUE)


mod_ls = list()
for(i in 1:length(mod_vec)){
  method = mod_vec[i]
  if(method == "lm" | method == "glmnet"){
    pre_process = c("center", "scale", "nzv", "medianImpute")
  }else{
    pre_process = "medianImpute"
  }
  mod = train(`work hours` ~ .,
              data = gss_tbl_train,
              method = method,
              metric = "Rsquared",
              na.action = na.pass,
              trControl = reuseControl,
              preProcess = pre_process)
  mod_ls[[i]] = mod
}

results = function(train_mod){
  algo = train_mod$method
  cv_rsq = str_remove(format(round(max(train_mod$results$Rsquared), 2), nsmall = 2), "^\\d")
  preds = predict(train_mod, gss_tbl_test, na.action = na.pass)
  ho_rsq = str_remove(format(round(cor(preds, gss_tbl_test$`work hours`)^2, 2), nsmall = 2), "^\\d")
  return(c("algo" = algo, "cv_rsq" = cv_rsq, "ho_sq" = ho_rsq))
}

table1_tbl <- as_tibble(t(sapply(mod_ls, results)))

# The results varied significantly between models, with tree-based methods and regularized regression outperforming the linear model due to their ability to capture complex, non-linear patterns in the data. The linear model's underperformance likely stems from its inability to model these complexities
# The R-squared values decreased from k-fold CV to holdout CV for all models, indicating a potential overfit to the training data and suggesting that the models may not generalize as well to unseen data. This drop is a common occurrence when models capture noise in the training data that does not represent true underlying patterns
# For a real-life prediction problem, I would choose the Random Forest model (ranger) due to its balance between high cv_rsq and relatively stable performance on the holdout set, indicating good generalization. While XGBoost showed the highest cv_rsq, its larger drop in performance on the holdout set suggests overfitting. The choice involves tradeoffs between interpretability and predictive accuracy, with Random Forest offering a middle ground 
