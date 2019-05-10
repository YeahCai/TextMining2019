# Classify Emails
# Ye (Kate) Cai
# 2019-04-26

######  Overview  #############################################################
# We have a data set of emails. We would like to classify "fake news" from real.
#   The data have raw features: text body, news title, and news source. The
#   response is "Real" or "Fake".

# install.packages("tidyverse")
library(tidyverse)
fakeNewsTrain_df <- read_csv("fake_news_train.csv")

# Explore
anyNA(fakeNewsTrain_df)
dim(fakeNewsTrain_df)


###  Goals  ###
# Build a model to discriminate real news from fake news, and also construct
#   visuals to aid in the model description to the masses.

# For an overview of text mining in R, see
#   https://www.tidytextmining.com/index.html


######  Pre-processing  #######################################################
# We need to transform our data matrix into "tidy text".
#install.packages("tidytext")
library(tidytext)

fakeNewsTrain_df <- 
  fakeNewsTrain_df %>% 
  mutate(ID = 1:nrow(fakeNewsTrain_df))


# Include the individual words from the title
splitTitles_df <- fakeNewsTrain_df %>% 
  unnest_tokens(word, title) %>% 
  mutate(textFromBody = FALSE) %>% 
  select(-text)

splitText_df <- fakeNewsTrain_df %>% 
  unnest_tokens(word, text) %>% 
  mutate(textFromBody = TRUE) %>% 
  select(-title)

splitFakeNews_df <- bind_rows(
  splitText_df,
  splitTitles_df
) %>% 
  arrange(ID)


######  Clean the Text  #######################################################
# adding stop words
View(stop_words)

# we found 12 words to add to the stop_words table (this is just an example)
numWords <- 12 
myStopWords_df <- tibble(
  word = c(
    "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself",
    "she", "her", "hers", "herself"
  ),
  lexicon = rep("ye", numWords)
)

# now add them
myStopWords_df <- bind_rows(
  stop_words,
  myStopWords_df
)


# Remove stop words
cleanSplitFN_df <- anti_join(
  splitFakeNews_df,
  myStopWords_df,
  by = "word"
)



###  Clean Bad Entries  ###
# We have some entries that are gibberish alphanumerics
cleanSplitFN_df %>%
  select(word) %>%
  arrange(word) %>%
  View()

# Split off rows with numbers
cleanSplitFN_df <- cleanSplitFN_df %>% 
  filter(!str_detect(word, "[0-9]"))
# This removes websites with numbers in them, so not ideal



######  Explore Data  #########################################################
# Plot most common words in body
cleanSplitFN_df %>% 
  filter(textFromBody == TRUE) %>% 
  group_by(label) %>% 
  count(word, sort = TRUE) %>% 
  slice(1:20) %>% 
  ggplot() +
    geom_col(aes(x = word, y = n)) +
    coord_flip() +
    ggtitle("Most Common Words in News Text Body") +
    facet_grid(~label)

# Plot most common words in title
cleanSplitFN_df %>% 
  filter(textFromBody == FALSE) %>% 
  group_by(label) %>% 
  count(word, sort = TRUE) %>% 
  slice(1:20) %>% 
  ggplot() +
  geom_col(aes(x = word, y = n)) +
  coord_flip() +
  ggtitle("Most Common Words in News Text Title") +
  facet_grid(~label)

# Find words that only exist in fake emails:
anti_join(
  cleanSplitFN_df %>% 
    filter(label == "Fake"),
  cleanSplitFN_df %>% 
    filter(label == "Real"),
  by = "word"
) %>% 
  count(word, sort = TRUE)

# Find words that only exist in real emails:
anti_join(
  cleanSplitFN_df %>% 
    filter(label == "Real"),
  cleanSplitFN_df %>% 
    filter(label == "Fake"),
  by = "word"
) %>% 
  count(word, sort = TRUE)


######  Remove Equally-occuring Words  ########################################
# Remove words with roughly the same chance of occurance in each type of news
#   article

summary(as.factor(cleanSplitFN_df$label))

# 17k counts, but we need 17k probabilities
cleanSplitFN_df %>% 
  group_by(label, textFromBody) %>% 
  count(word)

###  Split Data by Group  ###
# Split into four lists
cleanSplitFN2_df <- 
  cleanSplitFN_df %>% 
  mutate(textFrom = ifelse(textFromBody, "Body", "Title")) %>% 
  select(-textFromBody)

cleanSplitFN2_df <- 
  cleanSplitFN2_df %>% 
  mutate(group = paste0(label, "_x_", textFrom)) %>% 
  select(-label, -textFrom)

cleanFN_ls <- split(
  cleanSplitFN2_df,
  f = cleanSplitFN2_df$group
)

# Add Probabilities for Each Word
cleanWordProbs_ls <- 
  lapply(cleanFN_ls, function(df){
    
    nObs <- nrow(df)
    df %>% 
      count(word) %>% 
      # "n" is the column added by the count() function
      mutate(prob = n / nObs)
    
  })

###  Words with Low Discriminatory Power  ###
# Rows with words to keep:
inner_join(
  cleanWordProbs_ls$Fake_x_Body,
  cleanWordProbs_ls$Real_x_Body,
  by = "word"
) %>% 
  mutate(ratio = prob.x / prob.y) %>% 
  filter((ratio < 1 / 1.5) | (ratio > 1.5))
# This cuts from 3973 to 2047 words

# Words to Cut
lowPowerBodyWords_char <- 
  inner_join(
    cleanWordProbs_ls$Fake_x_Body,
    cleanWordProbs_ls$Real_x_Body,
    by = "word"
  ) %>% 
  mutate(ratio = prob.x / prob.y) %>% 
  filter((ratio > 1 / 1.5) & (ratio < 1.5)) %>% 
  select(word) %>% 
  unlist() %>% 
  as.character()


# Rows with title words to keep:
inner_join(
  cleanWordProbs_ls$Fake_x_Title,
  cleanWordProbs_ls$Real_x_Title,
  by = "word"
) %>% 
  mutate(ratio = prob.x / prob.y) %>% 
  filter((ratio < 1 / 1.5) | (ratio > 1.5))
# This cuts from 155 to 77 words

# Words to Cut
lowPowerTitleWords_char <- 
  inner_join(
    cleanWordProbs_ls$Fake_x_Title,
    cleanWordProbs_ls$Real_x_Title,
    by = "word"
  ) %>% 
  mutate(ratio = prob.x / prob.y) %>% 
  filter((ratio > 1 / 1.5) & (ratio < 1.5)) %>% 
  select(word) %>% 
  unlist() %>% 
  as.character()


###  Join Data Frames and Filter Out LP Words ###
LPwords_char <- c(lowPowerBodyWords_char, lowPowerTitleWords_char)

# Start with 17321 unique words
cleanWordProbs_df <-
  cleanWordProbs_ls %>% 
  bind_rows(.id = "group") %>% 
  separate(group, c("label", "textFrom"), "_x_") %>% 
  filter(!(word %in% LPwords_char))
# Now we have 12916 words (with higher discriminatory power)

fakeNewsPowerWords_df <- cleanWordProbs_df %>% 
  select(-prob)

write_csv(fakeNewsPowerWords_df, "fake_news_WC_clean_train.csv")


###  Models  ###
# Kate has tried the topicmodels::LDA (Latent Dirichlet Allocation)
install.packages("topicmodels")
library(topicmodels)

fakeNews_dtm <- cast_dtm(
  data = fakeNewsPowerWords_df,
  document = textFrom,
  term = word,
  value = n
)

fakeNews_lda <- LDA(
  fakeNews_dtm,
  k = 2,
  control = list(seed = 1234)
)

tidy(fakeNews_lda, matrix = "beta")


###Modeling based on "text mining"
library(tm)
train_model <- read_csv("fake_news_train.csv")


# remove things before converting to corpus

x <- fake_news_train$text[6]
gsub('\x', '', x)
gsub('<>', '', x)


fake_news_train$text2 <- gsub('[[:punct:]]', '', fake_news_train$text)
fake_news_train$text3 <- gsub('<>', '', fake_news_train$text2)


require(caTools)
set.seed(123456)
#sample <-sample.split(train_model,SplitRatio = 0.75)
#dtrain <- subset(train_model,sample==TRUE)
#for text part
#create a corpus
corpusText = VCorpus(VectorSource(dtrain$text))
# Remove Punctuation
corpusText <- tm_map(corpusText, myFunction)
corpusText <- tm_map(corpusText, removePunctuation)
corpusText <- tm_map(corpusText,removeNumbers)
corpusText <- tm_map(corpusText, tolower)
# Remove Stopwords
corpusText = tm_map(corpusText,removeWords,stopwords("english"))
corpusText = tm_map(corpusText, stemDocument)
corpusText <- Corpus(VectorSource(corpusText))
# Create matrix
dtmText = DocumentTermMatrix(corpusText)
# Remove sparse terms
dtmText = removeSparseTerms(dtmText, .95)
# Create data frame
wordsText = as.data.frame(as.matrix(dtmText))
wordsText[1:6,1:6]
colnames(wordsText) = paste("Text_", colnames(wordsText))

##for title part
corpusTitle = VCorpus(VectorSource(train_model$title))
#corpusR = tm_map(corpusR, PlainTextDocument)
corpusTitle = tm_map(corpusTitle, removePunctuation)
corpusTitle <- tm_map(corpusTitle,removeNumbers)
corpusTitle <- tm_map(corpusTitle, tolower)
corpusTitle = tm_map(corpusTitle, removeWords, 
                 stopwords("english"))
corpusTitle = tm_map(corpusTitle, stemDocument)
dtmTitle = DocumentTermMatrix(corpusTitle)
# Remove sparse terms
dtmTitle = removeSparseTerms(dtmTitle, .95)
# Create data frame
wordsTitle = as.data.frame(as.matrix(dtmTitle))
wordsTitle[1:6,1:6]
colnames(wordsTitle) = paste("Title_", colnames(wordsTitle))


##combine them together
wikiWords = cbind(wordsAdded, wordsRemoved)


###try another function from "tidtext" package

