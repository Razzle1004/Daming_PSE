
#set library
library(dplyr)
library(plotly)
library(keras)
library(kerasR)
library(BBmisc)

setwd("C:/Users/Razzfield/Desktop/Daming Kuliah-Prak/Project/Daming_PSE")
df <- read.csv("C:/Users/Razzfield/Desktop/Daming Kuliah-Prak/Project/Kelompok01_data_bersih.csv", row.names=1)
AC <- df %>%
  filter(symbol =="AC")
GLO <- df %>%
  filter(symbol=="GLO")
SM <- df%>%
  filter(symbol=="SM")

AC$date <- as.Date(AC$date)
GLO$date <- as.Date(GLO$date)
SM$date <- as.Date(SM$date)

plot_ly(AC, x = ~date, y = ~close.price, type = "scatter", mode = "markers", color = ~trading.volume) %>%
    layout (title="AC Stock Closing Price")
acf(AC$close.price, lag.max = 1000)


AC=normalize(AC,"range",c(0,1))

datalags = 5
train = AC[seq(1800 + datalags), ]
test = AC[1800 + datalags + seq(630 + datalags), ]
batch.size = 5

x.train = array(data = lag(cbind(train$close.price, train$trading.volume), datalags)[-(1:datalags), ], dim = c(nrow(train) - datalags, datalags, 2))
y.train = array(data = train$close.price[-(1:datalags)], dim = c(nrow(train)-datalags, 1))

x.test = array(data = lag(cbind(test$trading.volume, test$close.price), datalags)[-(1:datalags), ], dim = c(nrow(test) - datalags, datalags, 2))
y.test = array(data = test$close.price[-(1:datalags)], dim = c(nrow(test) - datalags, 1))

model <- keras_model_sequential()

model %>%
  layer_lstm(units = 25,
             input_shape = c(datalags, 3),
             return_sequences = TRUE,
             stateful = TRUE,
             batch_size= batch.size) %>%
  time_distributed(layer_dense(units=3)) %>%
  compile(loss = 'mean_squared_error', optimizer = 'adam')

model

for(i in 1:50){
  model %>% fit(x = x.train,
                y = y.train,
                batch_size = batch.size,
                epochs = 4
                )
  model %>% reset_states()
}

pred_train <- model %>% predict(x.train,batch_size=batch.size)
pred_test <- model %>% predict(x.test, batch_size = batch.size)


plot_ly(AC, x = ~date, y = ~close.price, type = "scatter", mode = "markers", color = ~trading.volume) %>%
  add_trace(y = c(rep(NA, 1800), pred_out), x = AC$date, name = "LSTM prediction", mode = "lines") %>%
  layout (title="AC Stock Closing Price (masih berantakan,belum tuning)")

plot(x = y.test, y = pred_out)
