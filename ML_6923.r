#Load the data
library(readr)
library(plyr)
library(dplyr)
library(tidyverse)
library(usethis)
library(devtools)
#install_github("vqv/ggbiplot", force=TRUE)
library(grid)
library(ggbiplot)
library(ggplot2)
library(lattice)
library(caret)
#From IBM Terminal
df <- read_csv("/home/2021/nyu/fall/ap5254/hw01/NYPD_Complaint_Data_Current__Year_To_Date_.csv")
#If you have the file
#df <- read_csv("C:/Users/poona/Downloads/NYPD_Complaint_Data_Current__Year_To_Date_.csv")
#If you have the link
#df = read.csv(url("https://data.cityofnewyork.us/api/views/5uac-w243/rows.csv?accessType=DOWNLOAD"))
#Check dimensions
dim(df)

#Change names to numbers to help reduce bias
names(df) = c(1:36)
names(df)

#Label the dependent variable
names(df)[14] = 'Dependent'
names(df)

head(df)

#Imbalance check
sum(df$Dependent=='FELONY')/nrow(df)
sum(df$Dependent=='MISDEMEANOR')/nrow(df)
sum(df$Dependent=='VIOLATION')/nrow(df)

ggplot(data = df) +
  geom_bar(mapping = aes(Dependent))

#Get rid of Identifier
df = df[-c(1)]

#Data Types of each column
str(df)

#Removing columns that have missing values summing at least half of the total amount of observations
colSums(is.na(df))
nrow(df)/2
df = df[-c(5,6,8,9,16,22,26)]

#Removing columns that are a description of another column
df = df[-c(11,14)]
str(df)

#Principle Component Analysis
pca<- prcomp(df[,c(1,21,22,23,24)], center = TRUE,scale. = TRUE)
str(pca)
ggbiplot(pca,labels = df$`13`, ellipse=TRUE, groups=df$Dependent)
#91.2

pca<- prcomp(df[,c(1,21,22)], center = TRUE,scale. = TRUE)
str(pca)
ggbiplot(pca,labels = df$`13`, ellipse=TRUE, groups=df$Dependent)
#89.5

pca<- prcomp(df[,c(1,23,24)], center = TRUE,scale. = TRUE)
str(pca)
ggbiplot(pca,labels = df$`13`, ellipse=TRUE, groups=df$Dependent)
#89.5

pca<- prcomp(df[,c(1,8)], center = TRUE,scale. = TRUE)
str(pca)
ggbiplot(pca,labels = df$`13`, ellipse=TRUE, groups=df$Dependent)
#Removing column 8 as that seems to be a 100% correlated to the dependent variable
df = df[-c(8)]
df = df[-c(24,25)]

#randomForest and Variable Importance
library(randomForest)
df$Dependent = as.factor(df$Dependent)
table(df$Dependent)
colnames(df) = c('A','B','C','D','E','F','G','Dependent','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V')

df$B = as.factor(df$B)
df$C = as.factor(df$C)
df$E = as.factor(df$E)
df$G = as.factor(df$G)
df$H = as.factor(df$H)
df$I = as.factor(df$I)
df$K = as.factor(df$K)
df$L = as.factor(df$L)
df$M = as.factor(df$M)
df$N = as.factor(df$N)
df$O = as.factor(df$O)
df$P = as.factor(df$P)
df$Q = as.factor(df$Q)
df$R = as.factor(df$R)

df$C = NULL
df$D = NULL
df$K = NULL
df$L = NULL

set.seed(222)
index = sample(2, nrow(df), replace = TRUE, prob = c(0.7,0.3))
train = df[index==1,]
test = df[index==2,]
str(train)
summary(train)

rf = randomForest(Dependent~., train, na.action=na.omit)
rf

VI =varImp(rf)
varImpPlot(rf,main="Variable Importance")

str(df)
#VIF
library(car)
m = lm(A~F+J+S+T+U+V, data=df)
vif(m)
