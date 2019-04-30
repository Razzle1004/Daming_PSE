#set library
library(dplyr)
library(plotly)
library(keras)
library(kerasR)

setwd("C:/Users/Razzfield/Desktop/Daming Kuliah-Prak/Project")
df <- read.csv("Kelompok01_data_bersih.csv", row.names=1)
AC <- df %>%
  filter(symbol =="AC")
GLO <- df %>%
  filter(symbol=="GLO")
SM <- df%>%
  filter(symbol=="SM")

AC.y = AC$close.price
AC$date <- as.Date(AC$date)
GLO$date <- as.Date(GLO$date)
SM$date <- as.Date(SM$date)

plot_ly(SM, x = ~date, y = ~close.price, type = "scatter", mode = "markers", color = ~trading.volume) %>%
  layout (title="SM Stock Closing Price")
acf(SM$close.price)

msd.price = c(mean(SM$close.price), sd(SM$close.price))
msd.vol = c(mean(SM$trading.volume), sd(SM$trading.volume))
SM$close.price = (SM$close.price - msd.price[1])/msd.price[2]
SM$trading.volume = (SM$trading.volume - msd.vol[1])/msd.vol[2]
summary(SM)

datalags = 3
train = SM[seq(2100 + datalags), ]
test = SM[2100 + datalags + seq(329 + datalags), ]
batch.size = 7
acf(train$close.price)

x.train = array(data = lag(cbind(train$close.price, train$trading.volume), datalags)[-(1:datalags), ], dim = c(nrow(train) - datalags, datalags, 2))
y.train = array(data = train$close.price[-(1:datalags)], dim = c(nrow(train)-datalags, 1))

x.test = array(data = lag(cbind(test$close.price, test$trading.volume), datalags)[-(1:datalags), ], dim = c(nrow(test) - datalags, datalags, 2))
y.test = array(data = test$close.price[-(1:datalags)], dim = c(nrow(test) - datalags, 1))

model <- keras_model_sequential()

model %>%
  layer_lstm(units = 25,
             input_shape = c(datalags, 2),
             batch_size = batch.size,
             return_sequences = TRUE,
             stateful = TRUE) %>%
  layer_dropout(rate = 0.00001) %>%
  layer_lstm(units = 5,
             return_sequences = FALSE,
             stateful = TRUE) %>%
  layer_dropout(rate = 0.001) %>%
  layer_dense(units = 1) %>%
  summary()

model %>%
  compile(loss = 'mae', optimizer = 'adam')

model

for(i in 1:4){
  model %>% fit(x = x.train,
                y = y.train,
                batch_size = batch.size,
                epochs = 100,
                verbose = 1,
                shuffle = FALSE)
  model %>% reset_states()
}

pred_out <- model %>% predict(x.test, batch_size = batch.size) %>% .[,1]


plot_ly(SM, x = ~date, y = ~close.price, type = "scatter", mode = "markers", color = ~trading.volume) %>%
  add_trace(y = c(rep(NA, 2100), pred_out), x = SM$date, name = "LSTM prediction", mode = "lines") %>%
  layout (title="SM Stock Closing Price")

plot(x = y.test, y = pred_out)
