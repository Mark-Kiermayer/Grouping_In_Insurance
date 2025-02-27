#install.packages("MortalityTables")
library(scales)
library(ggplot2)
library(MortalityTables)

mortalityTables.load("Germany_Endowments_DAV2008T")

# SUSM death probs
A = 0.00022
B = 2.7*10^(-6)
c = 1.124
x = seq(from=0, to=125, by = 1)
y = 1-exp(-A-B/log(c)*c^x*(c-1))

data_male = MortalityTables::deathProbabilities(DAV2008T.male)
data_female = MortalityTables::deathProbabilities(DAV2008T.female)

write.csv(data_male,"DAV2008Tmale.csv", row.names = FALSE)


plot(data_female, type = 'l', log= 'y', xlab = 'age', ylab = 'probability', 
     cex.lab = 1, cex.axis = 1,lwd = 2, col = 'grey')
lines(data_male, type = 'l', lwd = 2, col = 'black')
lines(x,y, type = 'l', lty = 2, lwd = 2)
legend(1,c('DAV 2008T (male)','DAV 2008T (female)','SUSM'), col = c('black', 'grey', 'black'), lty = c(1,1,2), cex = 1)

#print(data_male)

