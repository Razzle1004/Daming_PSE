
#set library
library(dplyr)
library(plotly)
library(keras)
library(kerasR)

setwd("C:/Users/Razzfield/Desktop/Daming Kuliah-Prak/Project")
df <- read.csv("C:/Users/Razzfield/Desktop/Daming Kuliah-Prak/Project/Kelompok01_data_bersih.csv", row.names=1)
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

plot_ly(AC, x = ~date, y = ~close.price, type = "scatter", mode = "markers", color = ~trading.volume) %>%
    layout (title="AC Stock Closing Price")
acf(AC$close.price, lag.max = 1000)

msd.price = c(mean(AC$close.price), sd(AC$close.price))
msd.vol = c(mean(AC$trading.volume), sd(AC$trading.volume))
AC$close.price = (AC$close.price - msd.price[1])/msd.price[2]
AC$trading.volume = (AC$trading.volume - msd.vol[1])/msd.vol[2]
summary()

datalags = 10
train = AC[seq(1800 + datalags), ]
test = AC[1800 + datalags + seq(630 + datalags), ]
batch.size = 15

x.train = array(data = lag(cbind(train$close.price, train$trading.volume), datalags)[-(1:datalags), ], dim = c(nrow(train) - datalags, datalags, 2))
y.train = array(data = train$close.price[-(1:datalags)], dim = c(nrow(train)-datalags, 1))

x.test = array(data = lag(cbind(test$trading.volume, test$close.price), datalags)[-(1:datalags), ], dim = c(nrow(test) - datalags, datalags, 2))
y.test = array(data = test$close.price[-(1:datalags)], dim = c(nrow(test) - datalags, 1))

model <- keras_model_sequential()

model %>%
  layer_lstm(units = 25,
             input_shape = c(datalags, 2),
             batch_size = batch.size,
             return_sequences = TRUE,
             stateful = TRUE) %>%
  layer_dropout(rate = 0.1) %>%
  layer_lstm(units = 4,
             return_sequences = FALSE,
             stateful = TRUE) %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 1)

model %>%
  compile(loss = 'mae', optimizer = 'adam')

model

for(i in 1:500){
  model %>% fit(x = x.train,
                y = y.train,
                batch_size = batch.size,
                epochs = 1,
                verbose = 0,
                shuffle = FALSE)
  model %>% reset_states()
}

pred_out <- model %>% predict(x.test, batch_size = batch.size) %>% .[,1]


plot_ly(AC, x = ~date, y = ~close.price, type = "scatter", mode = "markers", color = ~trading.volume) %>%
  add_trace(y = c(rep(NA, 1800), pred_out), x = AC$date, name = "LSTM prediction", mode = "lines") %>%
  layout (title="AC Stock Closing Price (masih berantakan,belum tuning)")

plot(x = y.test, y = pred_out)
