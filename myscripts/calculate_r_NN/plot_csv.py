import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

csv_input = pd.read_csv(filepath_or_buffer="D:\\kenkyu\\mine\\mitsuba2\\myscripts\\train_data\\NN_loss_save\\loss_record_f_0.0001_Adam.csv", encoding="ms932", sep=",")
loss1 = csv_input["loss"]
csv_input = pd.read_csv(filepath_or_buffer="D:\\kenkyu\\mine\\mitsuba2\\myscripts\\train_data\\NN_loss_save\\loss_record_f_0.001_Adam.csv", encoding="ms932", sep=",")
loss2 = csv_input["loss"]

csv_input = pd.read_csv(filepath_or_buffer="D:\\kenkyu\\mine\\mitsuba2\\myscripts\\train_data\\NN_loss_save\\loss_record_test_0.0001_Adam.csv", encoding="ms932", sep=",")
loss_test1 = csv_input["loss"]
csv_input = pd.read_csv(filepath_or_buffer="D:\\kenkyu\\mine\\mitsuba2\\myscripts\\train_data\\NN_loss_save\\loss_record_test_0.001_Adam.csv", encoding="ms932", sep=",")
loss_test2 = csv_input["loss"]


def moving_avg(in_x, in_y):
    np_y_conv = np.convolve(in_y, np.ones(15)/float(15), mode='valid') # 畳み込む
    out_x_dat = np.linspace(np.min(in_x), np.max(in_x), np.size(np_y_conv))

    return out_x_dat, np_y_conv

epochs = [i+1 for i in range(600)]

x1,y1 = moving_avg(epochs, loss1)
x2,y2 = moving_avg(epochs, loss2)
x3,y3 = moving_avg(epochs, loss_test1)
x4,y4 = moving_avg(epochs, loss_test2)

plt.rcParams["font.size"] = 18
plt.title("Loss function")
plt.xlabel("epochs")
plt.ylabel("Loss")
plt.plot(x1, y1, color='purple', label="Adam lr=1e-4")
plt.plot(x2, y2, color='g', label="Adam lr=1e-3")
plt.legend()
plt.xlim([0,500])
plt.show()

plt.title("Loss function")
plt.xlabel("epochs")
plt.ylabel("Loss")
plt.plot(x3, y3, color='purple', label="Adam lr=1e-4")
plt.plot(x4, y4, color='g', label="Adam lr=1e-3")
plt.legend()
plt.xlim([0,300])
plt.show()