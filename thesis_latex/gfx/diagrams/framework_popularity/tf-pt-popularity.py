import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
colors = ['#254167', '#ee5679', '#d9d874', '#9db0bf']

mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colors)

mentions_tf = np.array([228, 266])
mentions_pt = np.array([87, 252])
mentions_k  = np.array([42, 56])
years       = np.array([2018, 2019])
width       = 0.1

f, ax = plt.subplots(figsize=(6, 6))
ax.bar(years - 0.1, mentions_tf, label='TensorFlow', width=width)
ax.bar(years, mentions_pt, label='PyTorch', width=width)
ax.bar(years + 0.1, mentions_k, label='Keras', width=width)
ax.set_xticks(years)
ax.set_xticklabels(years)
ax.set_title('Mentions in ICLR submissions for Deep Learning Frameworks')
ax.set_xlim([2017, 2020])
ax.legend()

plt.savefig('popularity.pdf')
