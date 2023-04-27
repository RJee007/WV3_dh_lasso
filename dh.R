# Here, the main model is formulated using lasso techniques

library(leaps) # for regsubsets
library(glmnet)
library(corrplot)
library(caret)

RMSE <- function(pred, actual){
  return ( sqrt(mean((pred-actual)^2)) )
}

relRMSE <- function(pred, actual){
  return (sqrt(mean((pred-actual)^2) )/( max(actual)-min(actual)) )
}

get_model_formula <- function(id, object, outcome){
  # get models data
  models <- summary(object)$which[id,-1]
  # Get outcome variable
  #form <- as.formula(object$call[[2]])
  #outcome <- all.vars(form)[1]
  # Get model predictors
  predictors <- names(which(models == TRUE))
  predictors <- paste(predictors, collapse = "+")
  # Build model formula
  as.formula(paste0(outcome, "~", predictors))
}

######################## FORMULATE THE MAIN MODEL ##########################################
# This uses all plots available

df <- read.csv("data.csv")
# df_lm: dataframe for lm (regression)
# # Just the predictor variables
x_vars <- df[, !names(df) %in% c('dh')]
y_var <- df$dh

# As per the lasso tutorial: see notes
lambda_seq <- 10^seq(2, -2, by = -0.1)

set.seed(1) # for repeatability
cv_output <- cv.glmnet(data.matrix(x_vars), y_var, alpha = 1, lambda = lambda_seq, nfolds = 5)
lasso_best <- glmnet(x_vars, y_var, alpha = 1, lambda = cv_output$lambda.min)
cl <- coef(lasso_best)
cl2 <- as.data.frame(cl[cl[,1]!=0,])

# vl: variable list. The predictor (x) vars selected by glmnet
vl <- as.list(rownames(cl2))
vl <- vl[vl != "(Intercept)"] # remove "(Intercept)"
# df_lm: make a dataframe suitable for lm
# from df, select the following columns: dh + (all variabes in 'vl')
df_lm <- df[, names(df) %in% c('dh', vl)]

models <- regsubsets(dh~., data = df_lm, nvmax = 3)
# To print the selected predictors and their coefficients 
print(coef(models, 3))
model_form <- get_model_formula(3, models, "dh")
# Output is:
# dh ~ eigen_medium_stG18 + msGreen_ws13_ngl2_sh1_gv_AVG + pan_ws13_ngl30_sh1_gcorr_SD
print (model_form)

mod1 <- lm(dh ~ eigen_medium_stG18 + msGreen_ws13_ngl2_sh1_gv_AVG + pan_ws13_ngl30_sh1_gcorr_SD, data=df_lm)
summary(mod1)
# check lm assumptions
plot(mod1, 1) # residuals vs fitted                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
plot(mod1, 3) # 

# Are they correlated? No, max. is 0.29
df_3vars <- df_lm[,c("eigen_medium_stG18", "msGreen_ws13_ngl2_sh1_gv_AVG", "pan_ws13_ngl30_sh1_gcorr_SD")]
corrplot(cor(df_3vars), method="number")

# Examine the correlation of each of them to dh
df_4vars <- df_lm[,c("dh", "eigen_medium_stG18", "msGreen_ws13_ngl2_sh1_gv_AVG", "pan_ws13_ngl30_sh1_gcorr_SD")]
# examine the correlation...
cor_df <- as.data.frame(cor(df_4vars[-1], df_4vars$dh))
colnames(cor_df)[1] = "val"
cor_df$abs_val <- abs(cor_df$val) # this can be examined...


#################### NOW, THE CROSS VALIDATION PART ##################################

# dataframe for predicted and actual values
df_pav <- NULL

# The main loop of the loocv: each time, drop a row, form
# a model with all other rows and test on the dropped row.
# This is essentially the code above, but in a loop
for (cr in 1:nrow(df)) { # cr: current row
  print(cr)
  # df_crd: dataframe, current row dropped
  df_crd <- df[-cr,]
  # df_tr: dataframe, test row (only). Just 1 row. Used for testing
  df_tr <- df[cr,]
  
  x_vars <- df_crd[, !names(df_crd) %in% c('dh')]
  y_var <- df_crd$dh
  set.seed(1)
  cv_output <- cv.glmnet(data.matrix(x_vars), y_var, alpha = 1, lambda = lambda_seq, nfolds = 5)
  lasso_best <- glmnet(x_vars, y_var, alpha = 1, lambda = cv_output$lambda.min)
  cl <- coef(lasso_best)
  cl2 <- as.data.frame(cl[cl[,1]!=0,])
  seed_val <- 2
  # the number of selected predictor variables should be between 10 and 40. If not, try again.
  # Num. selected predictors: nrow(cl2)-1.
  # Also see: https://stats.stackexchange.com/questions/97777/variablity-in-cv-glmnet-results
  while ( ((nrow(cl2)-1)<10) | ((nrow(cl2)-1)>40) )  {
    print (paste("Trying with seed value: ", toString(seed_val)))
    set.seed(seed_val)  
    cv_output <- cv.glmnet(data.matrix(x_vars), y_var, alpha = 1, lambda = lambda_seq, nfolds = 5)
    lasso_best <- glmnet(x_vars, y_var, alpha = 1, lambda = cv_output$lambda.min)
    cl <- coef(lasso_best)
    cl2 <- as.data.frame(cl[cl[,1]!=0,])
    seed_val <- seed_val+1
  }
  print (paste("Num. variables chosen by glmnet: ", toString(nrow(cl2))))
  
  # vl: variable list. The independent vars selected by glmnet
  vl <- as.list(rownames(cl2))
  vl <- vl[vl != "(Intercept)"] # remove "(Intercept)"
  df_lm <- df[, names(df) %in% c('dh', vl)]
  
  models <- regsubsets(dh~., data = df_lm, nvmax = 3)
  
  # To print the selected predictors and their coefficients 
  # print(coef(models, 3))
  model_form <- get_model_formula(3, models, "dh")
  print (model_form)
  curr_mod <- lm(model_form, data = df_lm)
  # Use the lm object to predict dh for the test row (df_tr)
  # pv: predicted value
  pv <- predict(curr_mod, newdata=df_tr)
  actual <- c(df_tr[1,]$dh)
  predicted <- c(pv)
  df_line <- data.frame(predicted, actual)
  if (is.null(df_pav)){ # first time
    df_pav <- df_line
  } else {
    df_pav <- rbind(df_pav, df_line) # append the new line
  }
}

RMSE(df_pav$predicted, df_pav$actual) # 1.415412
relRMSE(df_pav$predicted, df_pav$actual) # 0.1205338

# Make a plot of predicted vs actual
plot <- ggplot(df_pav, aes(x=predicted, y=actual))+geom_point()+xlim(15,30)+ylim(15,30)
plot <- plot + xlab("Predicted dominant ht. (m)") + ylab("Actual dominant ht. (m)")
plot <- plot + geom_abline(slope=1, intercept=0, colour='red')
plot



