import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

csv_input = pd.read_csv(filepath_or_buffer="D:\\kenkyu\\VAE\\kenkyutest\\loss_record.csv", encoding="ms932", sep=",")

loss1 = csv_input["loss"]
csv_input = pd.read_csv(filepath_or_buffer="D:\\kenkyu\\VAE\\kenkyutest\\loss_record_f_0.0001_Adam.csv", encoding="ms932", sep=",")
loss2 = csv_input["loss"]

def moving_avg(in_x, in_y):
    np_y_conv = np.convolve(in_y, np.ones(3)/float(3), mode='valid') # 畳み込む
    out_x_dat = np.linspace(np.min(in_x), np.max(in_x), np.size(np_y_conv))

    return out_x_dat, np_y_conv

epochs1 = [i+1 for i in range(1000)]

x1,y1 = moving_avg(epochs1, loss1)
epochs2 = [i+1 for i in range(600)]
x2,y2 = moving_avg(epochs2, loss2)

plt.title("Loss function")
plt.xlabel("epochs")
plt.ylabel("Loss")
plt.plot(x1, y1, color='r', label='linear')

plt.plot(x2, y2, color='g', label='linear')
plt.xlim([0,200])

plt.show()