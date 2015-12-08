## Kaglle score: 0.256283

require(formula.tools)
require(leaps)
require(ggplot2)

setwd("~/Documents/kaggle/Liberty_Mutual_Group_Property_Inspection_Prediction/dev/R/")
source(file="RegSubsetExplorer.R")

EncodeColumns <- function(df){
  for (col.name in names(df)){
    #print(col.name)
    if (class(df[,col.name]) == "factor"){
      df[,col.name] <- match(df[,col.name], LETTERS)
    }
  } 
  return(df)
}

#load train and test 
data.raw  <- read.csv("../data/raw/train.csv")
submit.raw   <- read.csv('../data/raw/test.csv')

#str(data.raw)
head(data.raw)
head(submit.raw)

columns.to.drop <- c('T1_V13' ,'T2_V7','T2_V10', 'T1_V10','T2_V8')
data.pre <- data.raw[, !(names(data.raw) %in% columns.to.drop)]

data.pre <- data.pre[data.pre['Hazard'] < 18, ]

cat("Dim data.pre")
print(dim(data.pre))

train.size <- 15000
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
submit.data <- EncodeColumns(submit.raw)

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

write.csv(data.frame(Id=submit.raw$Id, Hazard=submit.pred),"R_regsubset_dev20150826.csv",row.names=F, quote=FALSE)
                               
  