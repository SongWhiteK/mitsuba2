import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

csv_input = pd.read_csv(filepath_or_buffer="D:\\kenkyu\\mine\\mitsuba2\\myscripts\\train_data\\NN_loss_save\\NN_loss_save_best_0814\\loss_record_f_0.00031_Adam.csv", encoding="ms932", sep=",")
loss1 = csv_input["loss"]

csv_input = pd.read_csv(filepath_or_buffer="D:\\kenkyu\\mine\\mitsuba2\\myscripts\\train_data\\NN_loss_save\\NN_loss_save_best_0814\\loss_record_test_0.00031_Adam.csv", encoding="ms932", sep=",")
loss_test1 = csv_input["loss"]



def moving_avg(in_x, in_y):
    np_y_conv = np.convolve(in_y, np.ones(15)/float(15), mode='valid') # 畳み込む
    out_x_dat = np.linspace(np.min(in_x), np.max(in_x), np.size(np_y_conv))

    return out_x_dat, np_y_conv

epochs = [i+1 for i in range(1000)]

x1,y1 = moving_avg(epochs, loss1)
x3,y3 = moving_avg(epochs, loss_test1)

plt.rcParams["font.size"] = 30
plt.title("Loss function")
plt.xlabel("epochs")
plt.ylabel("Loss")
plt.plot(x1, y1, color='purple', label="smoothing")
plt.plot(epochs, loss1, color='purple', label="not smoothing",alpha=0.3)
plt.legend()
plt.xlim([0,200])
plt.show()

plt.title("Loss function")
plt.xlabel("epochs")
plt.ylabel("Loss")
plt.plot(x3, y3, color='purple', label="smoothing")
plt.plot(epochs, loss_test1, color='purple', label="not smoothing",alpha=0.3)
plt.legend()
plt.xlim([0,200])
plt.show()