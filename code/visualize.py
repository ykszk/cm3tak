#%%
import scipy.io as sio

connectome = sio.loadmat('data/connectome_scale1.mat')
# %%
sc = connectome['sc']
# %%
sc['number_of_fibers']

#%%
import matplotlib.pyplot as plt
plt.hist(sc['number_of_fibers'][0], bins=50)
plt.xlabel('Number of fibers')
plt.ylabel('Frequency')
plt.title('Distribution of Number of Fibers in Connectome')
plt.show()

#%% heatmap
import seaborn as sns
plt.figure(figsize=(10, 8))
sns.heatmap(sc['number_of_fibers'][0][0], cmap='viridis')
plt.title('Heatmap of Number of Fibers in Connectome')
plt.xlabel('Region Index')
plt.ylabel('Region Index')
plt.show()

#%%
sc['number_of_fibers'][0][0].shape

#%%
import networkx as nx
import pickle
# G = nx.read_gpickle('data/connectome_scale1.gpickle')
with open('data/connectome_scale1.gpickle', 'rb') as f:
    G = pickle.load(f)
# %%
G.number_of_nodes()
nx.draw(G, with_labels=True, node_size=500, node_color='lightblue', font_size=10)