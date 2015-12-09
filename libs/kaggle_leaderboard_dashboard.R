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
# * ****************************************************************

build.leaderboard.dashboard <- function(prop.inspection.lb){

    # Ploting
    hist(prop.inspection.lb$pvt.score,breaks=160,main = "Liberty Mutual Group: Property Inspection Prediction", xlab = "Private Normalized Gini", xlim = c(0.3,0.4))
    abline(v = 0.387957, col="red", lty = 4)   # Private score
    abline(v = 0.385060, col="blue", lty = 4)  # Best public score
    legend("topleft", legend = c("My private score", "My public score"), col=c("red","blue"), pch=c(19,19))

    # Saving the plot
    dev.copy(png,'figures/hist_private_scores.png',width=12,height=8,units="in",res=100)
    dev.off()

    # Comparative top 100
    top100.pvt.lb <- prop.inspection.lb [ prop.inspection.lb$pvt.idx<=100, ]
    top100.pvt.lb$pub.levels <- cut(top100.pvt.lb$pub.idx, breaks = c(0,25,50,100), labels = c("top25","pub rank: (25,50]","pub rank: (50,100]")) 
    top100.pvt.lb$pvt.levels <- cut(top100.pvt.lb$pvt.idx, breaks = c(0,25,50,100), labels = c("top25","prv rank: (25,50]","prv rank: (50,100]")) 

    top10.pvt.lb <- prop.inspection.lb[ prop.inspection.lb$pvt.idx<=10, ]
    winner <- prop.inspection.lb[ prop.inspection.lb$pvt.idx==1, ]
    me <- prop.inspection.lb[ prop.inspection.lb$pvt.idx==916, ]

    boxplot(top100.pvt.lb$drank ~ top100.pvt.lb$pub.levels, main="Liberty Mutual Group: Property Inspection Prediction Leaderboard Rank", xlab="Public Rank", ylab="Delta Rank: Private - Public")

    # Saving the plot
    dev.copy(png,'figures/boxplot_pub_scores_vs_drank.png',width=12,height=8,units="in",res=100)
    dev.off()

    plot(top100.pvt.lb$dscore, top100.pvt.lb$drank, main="Liberty Mutual Group: Property Inspection Prediction Leaderboard Rank", xlab="Score Improvement", ylab="Rank Improvement")
    abline(h = 0, col="blue", lty = 3); abline(v = 0, col="blue", lty = 3)
    points(top10.pvt.lb$dscore,top10.pvt.lb$drank, col=8, pch=19)
    points(winner$dscore,winner$drank, col="red", pch=19)
    points(me$dscore,me$drank, col="blue", pch=19)
    legend("bottomleft", legend = c("top 10", "Winner", "Me"), col=c(8,"red","blue"), pch=c(19,19))

    # Saving the plot
    dev.copy(png,'figures/scatter_drank_vs_dscore.png',width=12,height=8,units="in",res=100)
    dev.off()
}
