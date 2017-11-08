library(rpart)
library(rpart.plot)
library(caret)
library(e1071)
library(data.table)
library(randomForest)
library(xgboost)
library(caTools)
library(nnet)

path <- "C:\\Praneetha\\06-hackathons\\HackerEarth\\NetworkAttacks\\"
setwd(path)

train_df <- fread("train_data.csv")
score_df <- fread("test_data.csv")


str(train_df)
colnames(train_df)

# Find missing values - no missing values
apply(train_df,2,function(x){sum(is.na(x))})
apply(test_df,2,function(x){sum(is.na(x))})
colSums(is.na(train_df))


# check target class
train_df[,.N/nrow(train_df), target]
prop.table(table(train_df$target))

# check unique values in each column
apply(train_df,2,function(x){length(unique(x))})
apply(test_df,2,function(x){length(unique(x))})

name.vector <- colnames(train_df)
train_df <- make.names(train_df,feature.names=setdiff(names(train_df),"connection_id"))
valid.name.vector <- make.names(name.vector,unique = TRUE)
colnames(train_df) <- valid.name.vector
train_df$target <- relevel(train_df$target,ref = names(sort(table(train_df$target)))[1])
train_df$target <- as.factor(train_df$target)

levels(train_df$target) <- make.names(levels(train_df$target))

for (i in 20:43)
  train_df[,i] <- as.factor(train_df[,i])

idvars <- c("connection_id")
cformula <- as.formula(paste("target~", 
                    c(paste(setdiff(colnames(train_df),c("target",idvars)),collapse="+"))))

vec <- createDataPartition(train_df$target,p=0.7,list=FALSE)

train <- train_df[vec,]
test <- train_df[-vec,]


# rpart

model <- train(cformula,data=train[,2:43],
               method="rpart",
               metric="Kappa")

rpart.plot(model$finalModel)
  
pred <- predict(model,newdata=test[,-c(1,43)])
table(pred)
confusionMatrix(pred,test$target)

# Random Forest

tcontrol <- trainControl(method = "repeatedcv",
                               number = 2,
                               repeats = 2,
                               classProbs = TRUE,
                         allowParallel = TRUE,
                         verboseIter = TRUE)


model_rf <- train(cformula,data=train,
                  method="rf",
                  metric="kappa",
                  tuneLength=20,
                  nodesize=50,
                  tuneGrid = expand.grid(.mtry=c(7:15)),
                  trControl=tcontrol)



pred <- predict(model_rf,newdata=test[,-c(1,43)])
table(pred)
confusionMatrix(pred,test$target)

pred_scoredata <- predict(model_rf,newdata=test[,-1])

score_df <- cbind(score_df,pred_scoredata)
score_df$target <- ifelse(score_df$pred_scoredata=='X1',1,
                      ifelse(score_df$pred_scoredata=='X0',0,2))


final_submission <- score_df[,c(1,44)]

write.csv(file="submission1.csv",final_submission,row.names = FALSE)


# xgboost


train_xg <- xgb.DMatrix(data=as.matrix(train[,-c("connection_id","target"),with=F]),label=train$target)
test_xg <- xgb.DMatrix(data=as.matrix(test[,-c("connection_id","target"),with=F]),label=test$target)
score_xg <- xgb.DMatrix(data=as.matrix(score_df[,-c("connection_id"),with=F]))

getMulAcc <- function(pred, train_xg)
{
  label <- getinfo(train_xg, "label")
  acc <- mean(label == pred)
  return(list(metric = "maccuracy", value = acc))
}



params <- list(objective = 'multi:softmax',
               num_class = 3)

watchlist <- list('train' = train_xg, 'valid' = test_xg)

clf <- xgb.train(params
                 ,train_xg
                 ,1000
                 ,watchlist
                 ,feval = getMulAcc
                 ,print_every_n = 20
                 ,early_stopping_rounds = 30
                 ,maximize = T
)

pred_xg <- predict(clf,score_xg)

score_df_final <- cbind(score_df,pred_xg)
score_df_final <- score_df_final[,c(1,43)]
colnames(score_df_final) <- c("connection_id","target")

write.csv(file="submission2.csv",score_df_final,row.names = FALSE)


