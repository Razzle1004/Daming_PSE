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

plot_ly(GLO, x = ~date, y = ~close.price, type = "scatter", mode = "markers", color = ~trading.volume) %>%
  layout (title="GLO Stock Closing Price")
acf(GLO$close.price)

msd.price = c(mean(GLO$close.price), sd(GLO$close.price))
msd.vol = c(mean(GLO$trading.volume), sd(GLO$trading.volume))
GLO$close.price = (GLO$close.price - msd.price[1])/msd.price[2]
GLO$trading.volume = (GLO$trading.volume - msd.vol[1])/msd.vol[2]
summary(AC)

datalags = 3
train = GLO[seq(2100 + datalags), ]
test = GLO[2100 + datalags + seq(330 + datalags), ]
batch.size = 15
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
  layer_lstm(units = 4,
             return_sequences = FALSE,
             stateful = TRUE) %>%
  layer_dropout(rate = 0.001) %>%
  layer_dense(units = 1)

model %>%
  compile(loss = 'mae', optimizer = 'adam')

model

for(i in 1:4){
  model %>% fit(x = x.train,
                y = y.train,
                batch_size = batch.size,
                epochs = 50,
                verbose = 1,
                shuffle = FALSE)
  model %>% reset_states()
}

pred_out <- model %>% predict(x.test, batch_size = batch.size) %>% .[,1]


plot_ly(GLO, x = ~date, y = ~close.price, type = "scatter", mode = "markers", color = ~trading.volume) %>%
  add_trace(y = c(rep(NA, 2100), pred_out), x = GLO$date, name = "LSTM prediction", mode = "lines") %>%
  layout (title="GLO Stock Closing Price")

plot(x = y.test, y = pred_out)
