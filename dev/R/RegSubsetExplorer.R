RegsubsetExplorer <- function(train.db,test.db,reg.formula,nvmax,
                              nbest=1,really.big=FALSE,force.in=NULL){
  require(formula.tools)
  
  reg.dev <- regsubsets(reg.formula,
                        data = train.db, nvmax = nvmax,
                        method="forward",
                        nbest=nbest,really.big=really.big,force.in=force.in)
  
  reg.summary <- summary(reg.dev)
  
  number.of.models <- length(reg.summary$adjr2)
  
  cat("number of models: ", number.of.models,"\n")
  
  lhs.formula <- lhs(reg.formula)
  
  train.size <- nrow(train.db)
  test.size <- nrow(test.db)
  
  ## Model Selection by Cross-Validation
  test.rmse = rep(NA, number.of.models)
  train.rmse = rep(NA, number.of.models)
  for (i in 1:number.of.models) {
    ## cat("model: ",i,"\n")
    mi <- GetModelRegSubset(i,reg.dev,train.db,
                            reg.formula = reg.formula,
                            nvmax,nbest,really.big,
                            force.in)
    
    train.rmse[i] <- sqrt(mean(mi$residuals^2))
    predi <- predict(mi, test.db)
    resi <- predi - test.db[,as.character(lhs.formula)]
    test.rmse[i] <- sqrt(mean(resi^2))
  }
  
  ## plotings
  n <- seq(1:length(reg.summary$adjr2))
  data.view <- data.frame(n=n,
                          adjr2=reg.summary$adjr2,
                          cp=reg.summary$cp)
  
  p1 <- ggplot(data=data.view,aes(x=n,y=adjr2,label=n))
  p1 <- p1 + geom_point(size=4) + labs(title="adjr2")
  p1 <- p1 + geom_text(hjust=-0.2, vjust=-0.2,angle = 15,size=6,
                       show_guide=FALSE)
  
  print(p1)
  
  ymax <- max(append(train.rmse,test.rmse))*1.10
  ymin <- min(append(train.rmse,test.rmse))*0.90
  
  data.view2 <- data.frame(n=n,
                           train.rmse=train.rmse,
                           test.rmse=test.rmse)
  
  p4 <- ggplot(data=data.view2,aes(x=n,y=train.rmse,label=n))
  
  p4 <- p4 + geom_point(aes(color="Train"),size=3,show_guide=FALSE)
  p4 <- p4 + geom_line(size=1,linetype="dotdash",show_guide=FALSE) + ylim(ymin,ymax)
  p4 <- p4 + geom_text(hjust=-0.2, vjust=-0.2,angle = 15,size=6,
                       show_guide=FALSE)
  
  p4 <- p4 + geom_point(aes(x=n,y=test.rmse,color="Test"),size=3)
  p4 <- p4 + geom_line(aes(x=n,y=test.rmse,color="Test"),size=1,linetype="dotdash")
  p4 <- p4 + geom_text(aes(x=n,y=test.rmse,label=n),hjust=-0.2, vjust=-0.2,
                       angle = 15,size=6)
  
  p4 <- p4 + scale_colour_manual(name='',
                                 values=c('Train'='black', 'Test'='blue'))
  
  p4 <- p4 + labs(xlab="n",ylab="MSE",title="MSE: Train vs test")
  print(p4)
  
  return(reg.dev)
}

GetModelFormula <- function(reg.dev,reg.formula,model.k){
  require(formula.tools)
  
  
  ck <- coef(reg.dev, model.k)
  ck.names <- names(ck)
  n <- length(ck.names)
  
  ## cat("ck\n")
  ## print(ck)
  ## Building rhs formula
  lhs.formula <- lhs(reg.formula)
  ## cat("lhs\n")
  ## print(lhs.formula)
  rhs.formula <- paste(ck.names[2:n],collapse=" + ")
  ## cat("rhs\n")
  ## print(rhs.formula)
  model.formula <- paste(lhs.formula," ~ ",rhs.formula)
  
  ## cat("model formula\n")
  ## print(model.formula)
  
  return(model.formula)
}

GetModelRegSubset <- function( model.k,reg.dev,train.db,reg.formula,nvmax,
                               nbest,
                               really.big=FALSE,
                               force.in=NULL,
                               verbose=FALSE){
  
  require(ascii)
  ## reg.dev <- regsubsets(reg.formula,
  ##                       data = train.db, nvmax = nvmax,
  ##                       nbest=nbest,
  ##                       really.big=really.big,
  ##                       force.in=force.in)
  
  reg.summary <- summary(reg.dev)
  models.number <- length(reg.summary$adjr2)
  n=seq(1:models.number)
  
  ## cat("model.number: ", models.number)
  
  
  ## Get coef of desired model
  
  ck <- coef(reg.dev, model.k)
  coef.k <- data.frame(names=names(ck),
                       coefs=as.numeric(coef(reg.dev, model.k)))
  
  
  ## Building rhs formula
  
  model.formula <- GetModelFormula(reg.dev,reg.formula,model.k)
  
  names.list <- paste(names(coef(reg.dev,1)),collapse=" + ")
  for(k in 2:models.number){
    names.aux <- paste(names(coef(reg.dev,k)),collapse=" + ")
    names.list <- append(names.list,names.aux)
  }
  
  model.table <- data.frame(n=n,
                            adjr2=reg.summary$adjr2,
                            coef.names=names.list)
  
  if(verbose == TRUE){
    ## Print first 7 model in model tables
    cat("Print Model: ", model.k,"\n")
    print(ascii(coef.k,include.rownames=FALSE,digits=4),type = "org")
    
    ## Print first 7 model in model tables
    cat("Print Model: ", model.k," neighbors\n")
    start.model <- ifelse(model.k >1 ,
                          model.k - 1,
                          model.k)
    
    end.model <- ifelse(model.k == reg.dev$nvmax ,
                        model.k,
                        model.k + 1)
    
    
    print(ascii(model.table[ seq(model.k -1,model.k+1,by=1)
                             ,1:2],include.rownames=FALSE,digits=4),type = "org")
    
    cat("Printing model formula\n")
    print(model.formula)
  }
  
  lm.model <- lm(model.formula,data=train.db)
  
  return(lm.model)
}