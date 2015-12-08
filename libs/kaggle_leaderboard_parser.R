# * ****************************************************************
#   Programmer[s]: Leandro Fernandes
#   Company/Institution:  
#   email: leandroohf@gmail.com
#   Program: Property_Inspection_Prediction                           
#   Commentary: 1st Kaggle Competition                                
#   Date: December 01, 2015
#   
#   The author believes that share code and knowledge is awesome.
#   Feel free to share and modify this piece of code. But don't be
#   impolite and remember to cite the author and give him his credits.
#
#   Reference: Shameless Stolen (Adapt) from Jeff Hebert
#   https://rstudio-pubs-static.s3.amazonaws.com/29531_4b5b689e7adf4448a8d420e6b356397c.html
#
#   This program parse the Kaggle leaderboard page and get private score, public score,
#   kaggler team id and Delta 
# * ****************************************************************

#################################################################
# Packages and Functions to make life easier
#################################################################

library(RCurl, quietly = TRUE)
library(XML, quietly = TRUE)
library(plyr, quietly = TRUE)

leaderboard <- function(kaggle.competition.url.root){
    # kaggle.competition.url.root is the competition url
    # usage: leaderboard('http://www.kaggle.com/c/mlsp-2014-mri')
    # Modified from shakeup function published by David Thaler
    
    # Note that this function pulls the leaderboard status at the end of the competition
    # per release notes by Kaggle Admin Jeff Moser published at, 
    # http://www.kaggle.com/forums/t/827/reliving-the-leaderboards-of-the-past
    
    pub.url <- paste0(kaggle.competition.url.root,'/leaderboard/public')
    pvt.url <- paste0(kaggle.competition.url.root,'/leaderboard/private')
    
    pub.raw <- getURL(pub.url)
    pvt.raw <- getURL(pvt.url)
    
    pub.doc <- htmlTreeParse(pub.raw, useInternalNodes=TRUE)
    pvt.doc <- htmlTreeParse(pvt.raw, useInternalNodes=TRUE)
    
    pub.ids <- xpathSApply(pub.doc, '//tr[@id]/@id')
    pvt.ids <- xpathSApply(pvt.doc, '//tr[@id]/@id')

    pub.score <- as.numeric(xpathSApply(pub.doc, "//abbr[@class='score']", xmlValue))
    pvt.score <- as.numeric(xpathSApply(pvt.doc, "//abbr[@class='score']", xmlValue))
        
    n <- length(pub.ids)
    pub.df <- data.frame('id'=pub.ids, 'pub.idx'=1:n, 'pub.score' = pub.score)
    pvt.df <- data.frame('id'=pvt.ids, 'pvt.idx'=1:n, 'pvt.score' = pvt.score)
    all.df <- join(pub.df, pvt.df)
 
    all.df$drank <- all.df$pvt.idx - all.df$pub.idx
    all.df$dscore <- all.df$pvt.score - all.df$pub.score
    
    return(all.df)
}
