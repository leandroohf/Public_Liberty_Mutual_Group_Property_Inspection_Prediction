## Kaglle score: not submit

require(formula.tools)
require(leaps)
require(ggplot2)

require(xgboost)

setwd("~/Documents/kaggle/Liberty_Mutual_Group_Property_Inspection_Prediction/dev/R")
source(file="RegSubsetExplorer.R")

EncodeColumns <- function(df){
  for (col.name in names(df)){
    #print(col.name)
    if (class(df[,col.name]) == "factor"){
      uniq.values <- sort(unique(df[,col.name]))
      #print(uniq.values)
      if(uniq.values[1] == 'N' & uniq.values[2] == 'Y' &  length(uniq.values) == 2 ){
        #print("Binarizando")
        dummy.vec <- rep(1,nrow(df))
        dummy.vec[ df[,col.name] == 'N']   <- 0
        df[,col.name]  <- dummy.vec
      }else{
        df[,col.name] <- as.numeric(match(df[,col.name], LETTERS))
      }
    }
  } 
  return(df)
}

# build Gini functions for use in custom xgboost evaluation metric
GiniAuxFunc <- function(y.true, y.pred) {
  df = data.frame(y.true = y.true, y.pred = y.pred)
  df <- df[order(df$y.pred, decreasing = TRUE),]
  df$random = (1:nrow(df))/nrow(df)
  totalPos <- sum(df$y.true)
  df$cumPosFound <- cumsum(df$y.true) # this will store the cumulative number of positive examples found (used for computing "Model Lorentz")
  df$Lorentz <- df$cumPosFound / totalPos # this will store the cumulative proportion of positive examples found ("Model Lorentz")
  df$Gini <- df$Lorentz - df$random # will store Lorentz minus random
  return(sum(df$Gini))
}

Gini <- function( y.true, y.pred) {
  GiniAuxFunc(y.true, y.pred) / GiniAuxFunc(y.true, y.true)
}

#load train and test 
data.raw  <- read.csv("../..//data/raw/train.csv")
submit.raw   <- read.csv('../../data/raw/test.csv')

#str(data.raw)
head(data.raw)
head(submit.raw)

columns.to.drop <- c('Id','T1_V13' ,'T2_V7','T2_V10', 'T1_V10')
#columns.to.drop <- c('Id','T1_V13' ,'T2_V7','T2_V10', 'T1_V10','T2_V8','T1_V6','T1_V7','T2_V3','T2_V11','T2_V13')
data.pre <- data.raw[, !(names(data.raw) %in% columns.to.drop)]

data.pre$Hazard_group <- rep(1,nrow(data.pre))
data.pre[ data.pre$Hazard > 1, 'Hazard_group' ] <- 0
#data.pre[ data.pre$Hazard > 1 & data.pre$Hazard < 6, 'Hazard_group' ] <- 2 
#data.pre[ data.pre$Hazard > 5 & data.pre$Hazard < 21, 'Hazard_group' ] <- 3 
#data.pre[ data.pre$Hazard > 20, 'Hazard_group' ] <- 4

data.pre <- data.pre[ data.pre['Hazard'] < 70, ]
#data.pre <- data.pre[ data.pre['Hazard'] > 1 & data.pre['Hazard'] < 10, ]

y.var <- 'Hazard'

cat("Dim data.pre")
print(dim(data.pre))

data.pre  <- EncodeColumns(data.pre)
#val.data    <- EncodeColumns(val.data)
submit.data <- EncodeColumns(submit.raw)

train.size <- 5000
val.size   <- nrow(data.pre) - train.size

set.seed(13)
r <- sample(nrow(data.pre),train.size)

#train.data <- data.pre[r,]
#val.data  <- data.pre[-r,]

train.data <- data.pre[1:train.size,]
val.data  <- data.pre[train.size:nrow(data.pre),]

dim(val.data)
dim(train.data)
head(train.data)

reg.formula <- as.formula("Hazard ~ . -Hazard_group")

nvmax <- 20
nbest <- 3
really.big <- TRUE

reg.dev <- RegsubsetExplorer(train.data,val.data,reg.formula,nvmax)

# Best model is 19, 18 e 20 tambem sao bons
summary(GetModelRegSubset( 15,reg.dev,train.data,reg.formula,nvmax,nbest))

lm.model <- GetModelRegSubset( 14,reg.dev,train.data,reg.formula,nvmax,nbest)

train.pred <- predict(lm.model,train.data)
train.pred <- data.frame(Hazard=train.data[,y.var],model=train.pred)

val.pred <- predict(lm.model,val.data)
val.pred <- data.frame(Hazard=val.data[,y.var],model=val.pred)

submit.pred <- predict(lm.model,submit.data)

head(train.pred,15)
head(val.pred,15)
head(submit.pred,7)

train.label <- as.matrix(train.data[,y.var])
train.matrix <- as.matrix(train.data[,!(names(train.data) %in% c('Id','Hazard','Hazard_group')) ])
xgb.train <- xgb.DMatrix(data = train.matrix , label=train.label)

val.label <- as.matrix(val.data[,y.var])
val.matrix <- as.matrix(val.data[,!(names(val.data) %in% c('Id','Hazard','Hazard_group')) ])
xgb.val   <- xgb.DMatrix(data = val.matrix, label=val.label)

xgb.submit   <- xgb.DMatrix(data = as.matrix(submit.data))

evalGini <- function(preds,xgb.train){
  train.labels <- getinfo(xgb.train, "label")
  g <- Gini(train.labels,preds)
  return(list(metric="gini",value=g))
}

param <- list("objective" = "count:poisson", #"reg:linear",
              "eta" = 0.01,
              "min_child_weight" = 5,
              "subsample" = 0.80,
              "colsample_bytree" = 0.80,
              "scale_pos_weight" = 1.00,
              "silent" = 1,
              #"booster" = "gbtree",
              "max_depth" = 7,
              "seed" = 19)#,
              #"eval_metric" = evalGini)

set.seed <- 19
watchlist <- list(val=xgb.val,train=xgb.train)
xgb.model <- xgboost::xgb.train(param=param, data = xgb.train, nthread = 4, nround = 3000,
                      watchlist=watchlist,## eval_metric = "rmse",
                      verbose=1, print.every.n = 250, early.stop.round=4,maximize = FALSE)

print(paste("best: index:",xgb.model$bestInd, "train-metric:",xgb.model$bestScore))

pred <- predict(xgb.model, xgb.train)
train.pred <- data.frame(Hazard=train.label,model=pred)
train.rmse <- sqrt( mean( (train.label - pred)^2 ) )
train.gini <- Gini(train.label,pred)

pred <- predict(xgb.model, xgb.val)
val.pred <- data.frame(Hazard=val.label,model=pred)
val.rmse <- sqrt( mean( (val.label - pred)^2 ) )
val.gini <- Gini(val.label,pred)

print(paste("train-rmse:", train.rmse, "   val-rmse:", val.rmse))
print(paste("train-gini:", train.gini, "   val-gini:", val.gini))

cat("Showing train.pre:")
head(train.pred,21)
tail(train.pred,21)

cat("Showing val.pre:")
head(val.pred,11)
tail(val.pred,7)
head(val.pred[val.pred$Hazard > 3 & val.pred$Hazard < 5, ],21)

submit.pred <- predict(xgb.model, xgb.submit)

# importanceRaw <- xgb.importance(, model = bst, data = sparse_matrix, label = output_vector)
# 
# # Cleaning for better display
# importanceClean <- importanceRaw[,`:=`(Cover=NULL, Frequence=NULL)]
# 
# head(importanceClean)
# 
# xgb.plot.importance(importance_matrix = importanceRaw)

write.csv(data.frame(Id=submit.raw$Id, Hazard=submit.pred),"R_xgb_dev20150828_1.csv",row.names=F, quote=FALSE)
