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

#load train and test 
data.raw  <- read.csv("../..//data/raw/train.csv")
submit.raw   <- read.csv('../../data/raw/test.csv')

#str(data.raw)
head(data.raw)
head(submit.raw)

columns.to.drop <- c('T1_V13' ,'T2_V7','T2_V10', 'T1_V10','T2_V8')
data.pre <- data.raw[, !(names(data.raw) %in% columns.to.drop)]

data.pre <- data.pre[data.pre['Hazard'] < 70, ]

cat("Dim data.pre")
print(dim(data.pre))

train.size <- 11000
val.size   <- nrow(data.pre) - train.size

set.seed(13)
r <- sample(nrow(data.pre),train.size)

train.data <- data.pre[r,]
val.data  <- data.pre[-r,]

dim(val.data)
dim(train.data)
head(train.data)

train.data <- EncodeColumns(train.data)
val.data <- EncodeColumns(val.data)
submit.data <-EncodeColumns(submit.raw)

reg.formula <- as.formula("Hazard ~ .")

nvmax <- 32
nbest <- 3
really.big <- TRUE

reg.dev <- RegsubsetExplorer(train.data,val.data,reg.formula,nvmax)

# Best model is 19, 18 e 20 tambem sao bons
summary(GetModelRegSubset( 18,reg.dev,train.data,reg.formula,nvmax,nbest))

lm.model <- GetModelRegSubset( 18,reg.dev,train.data,reg.formula,nvmax,nbest)

val.pred <- predict(lm.model,val.data)
val.pred <- data.frame(Hazard=val.data[,'Hazard'],model=val.pred)

submit.pred <- predict(lm.model,submit.data)

head(val.pred,15)
head(submit.pred,7)

xgb.train <- xgb.DMatrix(data = as.matrix(train.data[,2:ncol(train.data)]), label=as.matrix(train.data[,'Hazard']))
xgb.val   <- xgb.DMatrix(data = as.matrix(val.data[,2:ncol(val.data)]), label=as.matrix(val.data[,'Hazard']))

xgb.submit   <- xgb.DMatrix(data = as.matrix(submit.data))

param <- list("objective" = "reg:linear",
              "eta" = 0.01,
              "min_child_weight" = 5,
              "subsample" = 0.80,
              "colsample_bytree" = 0.80,
              "scale_pos_weight" = 1.00,
              "silent" = 1,
              "booster" = "gbtree",
              "max_depth" = 9,
              "eval_metric" = "rmse")
              
watchlist <- list(train=xgb.train, val=xgb.val)
xgb.model <- xgboost(param=param, data = xgb.train, nthread = 4, nround = 500,
                      watchlist=watchlist, 
                      eval.metric = "poisson-nloglik", verbose=1)

label = getinfo(xgb.train, "label")
pred <- predict(xgb.model, xgb.train)
train.pred <- data.frame(Hazard=train.data[,'Hazard'],model=pred)
train.rmse <- sqrt( mean( (label - pred)^2 ) )

label = getinfo(xgb.val, "label")
pred <- predict(xgb.model, xgb.val)
val.pred <- data.frame(Hazard=val.data[,'Hazard'],model=pred)
val.rmse <- sqrt( mean( (label - pred)^2 ) )

print(paste("train-rmse:", train.rmse, "   val-rmse:", val.rmse))

cat("Showing train.pre:")
head(train.pred,21)
tail(train.pred,21)

cat("Showing val.pre:")
head(val.pred,11)
tail(val.pred,7)
head(val.pred[val.pred$Hazard > 30 & val.pred$Hazard < 40, ],21)

submit.pred <- predict(xgb.model, xgb.submit)

# importanceRaw <- xgb.importance(, model = bst, data = sparse_matrix, label = output_vector)
# 
# # Cleaning for better display
# importanceClean <- importanceRaw[,`:=`(Cover=NULL, Frequence=NULL)]
# 
# head(importanceClean)
# 
# xgb.plot.importance(importance_matrix = importanceRaw)

write.csv(data.frame(Id=submit.raw$Id, Hazard=submit.pred),"R_xgb_dev20150826_1.csv",row.names=F, quote=FALSE)
