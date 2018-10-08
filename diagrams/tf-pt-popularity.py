import matplotlib.pyplot as plt
import matplotlib as mpl
colors = ['#254167', '#ee5679', '#d9d874', '#9db0bf']

mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colors)

mentions_tf = [228, 266]
mentions_pt = [87, 252]
mentions_k = [42, 56]
years = [2018, 2019]

f, ax = plt.subplots(figsize=(6, 6))
ax.bar(years, mentions_tf, label='TensorFlow', width=0.5)
ax.bar(years, mentions_pt, label='PyTorch', width=0.5)
ax.bar(years, mentions_k, label='Keras', width=0.5)
ax.set_xticks(years)
ax.set_xticklabels(years)
ax.set_title('Mentions in ICLR submissions for Deep Learning Frameworks')
ax.set_xlim([2017, 2020])
ax.legend()

plt.savefig('popularity.pdf')
