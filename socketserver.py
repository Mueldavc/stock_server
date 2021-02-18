import numpy as np
from datetime import datetime
from keras.models import load_model
from os import path
import sys

sys.path.append(path.abspath(r'C:\Users\mueld\Documents\Python_Projects\AutoEncoder'))
from DataSet.DataSet import StockData_robo
import socket


def calcregr(msg=''):
    regressao = load_model(r'C:\Users\mueld\Documents\Python_Projects\Stock_1\model_linear.h5')
    autoencoder = load_model(r'C:\Users\mueld\Documents\Python_Projects\Stock_1\model_autoencoder.h5')
    stockdata = StockData_robo('WIN$', date=datetime.today(), timeframe=15)
    stockdata.data_final(n_in=17)
    train_x = stockdata.train_x
    train_x = train_x.reshape(*train_x.shape, 1)
    y_pred = autoencoder.predict(train_x)
    x_train = y_pred.reshape(y_pred.shape[0], -1)
    y_pred = regressao.predict(x_train)
    y_pred = np.argmax(y_pred).astype(str)
    now = datetime.now()
    time = now.strftime("%H:%M:%S")
    print(y_pred, ' ', time)
    print('venda' if y_pred == "0" else 'compra')
    return str(y_pred)


class SocketServer:
    def __init__(self, address='', port=9090):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.address = address
        self.port = port
        self.sock.bind((self.address, self.port))
        self.cummdata = ''
        self.conn = ''
        self.addr = ''

    def recvmsg(self):
        self.sock.listen(1)
        self.conn, self.addr = self.sock.accept()
        # print('connected to', self.addr)
        self.cummdata = ''

        while True:
            data = self.conn.recv(200)
            self.cummdata += data.decode("utf-8")
            if not data:
                break
            self.conn.send(bytes(calcregr(self.cummdata), "utf-8"))
            return self.cummdata

    def __del__(self):
        self.sock.close()


print('inicio')

serv = SocketServer('127.0.0.1', 9090)

while True:
    msg = serv.recvmsg()
