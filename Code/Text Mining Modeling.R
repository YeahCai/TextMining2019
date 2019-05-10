###Modeling based on "text mining"
library(data.table)
library(purrr)
library(RecordLinkage)
library(stringr)
library(tm)
library(caTools)
library(xgboost)
library(readr)
library(tidyverse)
library(tidytext)

#original datsaset
fakenews_train <- read_csv("fake_news_train.csv")  #316 rows
fakenews_test <- read_csv("fake_news_test.csv")  #106 rows

#I also add "ID" column in the training dataset
fakenews_test$ID <- NULL

fakenews_train <- as.data.table(fakenews_train)
fakenews_test <- as.data.table(fakenews_test)
#join data togethet
fakenews_test[,label :="None"]
fakenews_test <- fakenews_test[,c("label","text","title","source")]
fakenews_train_test <- rbindlist(list(fakenews_train,fakenews_test))
dim(fakenews_train_test)  # 422, combine successfully
#add a new column
fakenews_train_test <- 
  fakenews_train_test %>% 
  mutate(ID = 1:nrow(fakenews_train_test))
View(fakenews_train_test)
unique(fakenews_train_test$source)

#those codes can not run successfully
#create a corpus of text
fakenews_text_corpus <- VCorpus(VectorSource(fakenews_train_test$text))
#check first 4 documents
inspect(fakenews_text_corpus[1:4])
#see two exmaples
print(lapply(fakenews_text_corpus[1:2], as.character))
#remove punctuation
fakenews_text_corpus <- tm_map(fakenews_text_corpus,removePunctuation)
#remove number 
fakenews_text_corpus <- tm_map(fakenews_text_corpus,removeNumbers)
#remove whitespace
fakenews_text_corpus <- tm_map(fakenews_text_corpus,stripWhitespace)
#tolower
fakenews_text_corpus <- tm_map(fakenews_text_corpus,tolower)
#remove stopwords
#fakenews_text_corpus <- tm_map(fakenews_text_corpus,removeWords,stopwords('english'))
#dropword <- c("br",stopwords('english'))
#fakenews_text_corpus <- tm_map(fakenews_text_corpus,removeWords,dropword)
# CAN NOT DROP STOPWORDS, MAYBE WE CAN DROP LATER

fakenews_text_corpus <- tm_map(fakenews_text_corpus,PlainTextDocument)
#performing stemming 
fakenews_text_corpus <- tm_map(fakenews_text_corpus,stemDocument)
fakenews_text_corpus[[1]]$content
#convert to document term matrix

docterm_fakenews_text_corpus <- DocumentTermMatrix(fakenews_text_corpus)
#why this one didnot work???

###NEW WAYS TO DEAL WITH THE DATASET
textdata <- data.table(ID=rep(unlist(fakenews_train_test$ID),
                              lapply(fakenews_train_test$text,length)),
                       text=unlist(fakenews_train_test$text))
textdata_split <- textdata %>%
  unnest_tokens(word,text)
#delete stop words
cleantextdata_split <- anti_join(
  textdata_split,
  stop_words,
  by="word"
)
#remove numbers
cleantextdata_split <- cleantextdata_split %>% 
  filter(!str_detect(word, "[0-9]"))

#convert features to lower
cleantextdata_split1 <- as.data.table(cleantextdata_split) #this is very important
cleantextdata_split1[,word :=unlist(lapply(word,tolower))]
#calculate count for every word
cleantextdata_split1[,count :=.N, word]
dim(cleantextdata_split1)   # 97180 rows, 3 columns
#in order to reduce the size of data frame, I just keep features which occur 5 or more times
#cleantextdata_split2 <- cleantextdata_split1 %>%
  #filter(count >= 5)
#I decide to keep all words here, and see whether I need just keep some words having high happening frequency
#convert columns into table <br/>
cleantextdata_split_new <- dcast(data = cleantextdata_split1, formula = ID ~ word, fun.aggregate = length, value.var = "word")
dim(cleantextdata_split_new)
#419 rows, 16934 columns
write_csv(cleantextdata_split_new,"cleantextdata_split_new510.csv")



###deal with "title" column in the data set
titledata <- data.table(ID=rep(unlist(fakenews_train_test$ID),
                              lapply(fakenews_train_test$title,length)),
                       title=unlist(fakenews_train_test$title))
titledata_split <- titledata %>%
  unnest_tokens(word,title)
#delete stop words
cleantitledata_split <- anti_join(
  titledata_split,
  stop_words,
  by="word"
)
#remove numbers
cleantitledata_split <- cleantitledata_split %>% 
  filter(!str_detect(word, "[0-9]"))


#convert features to lower
cleantitledata_split1 <- as.data.table(cleantitledata_split) #this is very important
cleantitledata_split1[,word :=unlist(lapply(word,toupper))]
#calculate count for every word
cleantitledata_split1[,count :=.N, word]

#in order to reduce the size of data frame, I just keep features which occur 5 or more times

#I decide to keep all words here, and see whether I need just keep some words having high happening frequency
#convert columns into table <br/>
cleantitledata_split_new <- dcast(data = cleantitledata_split1, formula = ID ~ word, fun.aggregate = length, value.var = "word")
dim(cleantitledata_split_new)  #422 rows, 1672 columns
write_csv(cleantitledata_split_new,"cleantitledata_split_new510.csv")

#join those two data frames "text" and "title" together
text_title_table <- cleantextdata_split_new %>%
  left_join(cleantitledata_split_new,by="ID")
#there should be 422 rows and 18606 columns
dim(text_title_table)
#successfully join together
write_csv(text_title_table,"text_title_table510.csv")

#sum the two columns together if their lowercase letter and upper letter are same
dataset_title_text <- read_csv("text_title_table510.csv")
dim(dataset_title_text)
#419 rows, 18605 columns
order(colSums(dataset_title_text))
max(colSums(dataset_title_text[,-ID]))  #1435
#If i did not delete "ID" column, the maximum will be the sum of "ID" 
which.max(colSums(dataset_title_text[,-ID]))
#"trump" is the word that shows most time, it's better to remove
min(colSums(dataset_title_text)) #0

# i need to delete some columns due to large range
keepfeatures <- names(dataset_title_text)[colSums(dataset_title_text)>5]
#we need to preserve those words, so that we could predict new news atrticle fake or real.
saveRDS(keepfeatures,"keepfeatures.rds")
#readRDS("~/Desktop/Text Mining/Text Mining/Kate Cai/keepfeatures.rds")

keepdata <- dataset_title_text[,names(dataset_title_text) %in% keepfeatures]
dim(keepdata)  #419 rows, 3331 columns

dim(dataset_title_text)  #18605
#combine "source" and "label" columns in this data frame
d1 <- fakenews_train_test %>%
  select(ID,source,label)
dim(d1)
#422 rows, 3 columns


all_tables <- keepdata %>%
  left_join(d1,by="ID") #success!!!
dim(all_tables)
#419 rows, 3333 columns, successsfully
#deal with "source" columns 
unique(all_tables$source.y)

all_tables$source.y <- ifelse(all_tables$source.y=="BuzzFeed" ,1,0)
dim(all_tables)
unique(all_tables$source.y)
#successfully changed

write_csv(all_tables,"all_tables_done.csv")
#successfully cleaned

###Modeling 
##split the data set into train and test
data_one <- read_csv("all_tables_done.csv")
unique(data_one$label)
dim(data_one)
#419 rows, 3333 columns

#train dataset
train_one <- data_one %>%
  filter(label!="None")
unique(train_one$label)
dim(train_one)
#316 rows, 3333 columns

#NEW KNOWLEDGE
`+`(1,2)



#test dataset
test_one <- data_one %>%
  filter(label=="None")
unique(test_one$label)
dim(test_one)
#103 rows, 3333 columns
test_one <- as.data.table(test_one) #important
test_one <- test_one[,label :=NULL]
dim(test_one)

#103 rows, 3332 columns
#successfully down

library(wordcloud)
train_numeric <- train_one[,!names(train_one) %in% drops]
wordcloud(names(train_numeric), train_numeric, min.freq = 100, scale = c(6,.1), colors = brewer.pal(6, 'Dark2'))

#successfully split

###XGBoost
sp <- sample.split(Y=train_one$label,SplitRatio = 0.6)
#create data for xgboost
xg_val <- train_one[sp,]
#dim(xg_val)
ID <- train_one$ID
target <- train_one$label

xg_val_target <- target[sp]

#drop some columsn in xgb.matrix
dim(train_one)  # 316, 3333
drops <- c("ID","label")
d_train <- train_one[,!names(train_one) %in% drops]
dim(d_train)  #316, 3331

d_val <- xg_val[,!names(xg_val) %in% drops]
dim(d_val)  #190,3331
  
#d_train[, sapply(d_train, class) %in% c('character', 'factor')]
class(d_train)

d_train <- xgb.DMatrix(data = as.matrix(d_train),label=target)
d_val <- xgb.DMatrix(data = as.matrix(d_val),label=xg_val_target)
d_test <- xgb.DMatrix(data = as.matrix(test_one[,-c("ID"),with=FALSE]))

param <- list(booster="xgbTree",
              objective ="binary:logistic",
              eval_metric="error",
              num_class=2,
              eta=.5,
              gamma=1,
              max_depth=4,
              min_child_weight=100,
              subsample=.7,
              colsample_bytree=.5)

set.seed(123456)
watch <- list(val=d_val,train=d_train)
xgb2 <- xgb.train(data = d_train,params = param,
                   watchlist = watch,nrounds = 50,
                  print_every_n = 5)

xg_pred <- as.data.table(t(matrix(predict(xgb2,d_test),nrow = 2,
                                  ncol = nrow(d_test))))

colnames(xg_pred) <- c("Fake","Real")

xg_pred <- cbind(data.table(ID=test_one$ID),xg_pred)
fwrite(xg_pred,"xgb_textmining.csv")

###random forest model
library(randomForest)
#install.packages("caret")
library(caret)
library(e1071)
library(readr)
rf_train <- train_one
rf_test <- test_one
trControl <- trainControl(method = "cv",
                          number = 10,
                          search = "grid")

set.seed(123456)

tuneGrid <- expand.grid(.mtry=c(1:10))
rf_mtry <- train(label~.,
                 data = rf_train,
                 method="rf",
                 metric="Accuracy",
                 tuneGrid=tuneGrid,
                 trControl=trControl,
                 importance=TRUE,
                 nodesize=14,
                 ntree=100)
print(rf_mtry)
prediction <- predict(rf_mtry,newdata = rf_test)
varImp(rf_mtry)  # look at the feature importance 



#tune the parameters
#not finished yet
  store_maxtrees <- list()
for (ntree in c(250,300,350,400,450,500,550,600,650,800,1000,2000)) {
  set.seed(123456)
  rf_model <- train(Target~.,
                    data = rf_train,
                    method="rf",
                    metric="Accuracy",
                    trControl=trControl,
                    ntree=ntree)
  key <- toString(ntree)
  store_maxtrees[[key]] <- rf_model
}

print(rf_mtry)
result_tree <- resample(store_maxtrees)
summary(result_tree)
