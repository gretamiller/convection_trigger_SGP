"""

Train random forest to predict onset of convection

"""
#%%
from rf_functions import *
import shap
import pickle
import matplotlib as mpl
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
from sklearn.cluster import KMeans
from scipy.interpolate import make_interp_spline
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from scipy.ndimage import gaussian_filter1d

#%%
path = 'output/shaply_rf_6/'
nclusters = 6
make_folder(path)
SEED = 0

#read in dataset
d2004 = pd.read_csv('inputs/ARM_sgp_2004.csv')
d2005 = pd.read_csv('inputs/ARM_sgp_2005.csv')
d2006 = pd.read_csv('inputs/ARM_sgp_2006.csv')
d2007 = pd.read_csv('inputs/ARM_sgp_2007.csv')
d2008 = pd.read_csv('inputs/ARM_sgp_2008.csv')
d2009 = pd.read_csv('inputs/ARM_sgp_2009.csv')
d2010 = pd.read_csv('inputs/ARM_sgp_2010.csv')
d2011 = pd.read_csv('inputs/ARM_sgp_2011.csv')
d2012 = pd.read_csv('inputs/ARM_sgp_2012.csv')
d2013 = pd.read_csv('inputs/ARM_sgp_2013.csv')
d2014 = pd.read_csv('inputs/ARM_sgp_2014.csv')
d2015 = pd.read_csv('inputs/ARM_sgp_2015.csv')
d2016 = pd.read_csv('inputs/ARM_sgp_2016.csv')
d2017 = pd.read_csv('inputs/ARM_sgp_2017.csv')
d2018 = pd.read_csv('inputs/ARM_sgp_2018.csv')

d = pd.concat([d2004,d2005,d2006,d2007,d2008,d2009,d2010,d2011,d2012,d2013,d2014,d2015,d2016,d2017,d2018])
d = d.reset_index(drop=True)

#just use June, July and Aug for convective cases
d['timef'] = pd.to_datetime(d['time'])
summer = d[(d['timef'].dt.month >=6) & (d['timef'].dt.month <= 8)]

#define x and y
inputs = ['LH', 'SH', 'T_srf', 'rh',
       'dCAPE_zm_dilute_tzhang', 'CIN_mu', 'LCL',
       'diluteCAPE_zm_mu',
       'PW_q',
       'T_low', 'T_mid', 'T_high',
       'q_low', 'q_mid', 'q_high', 
       #'rh_low', 'rh_mid', 'rh_high', 
       'q_adv_h_low', 'q_adv_h_mid', 'q_adv_h_high', 
       's_adv_h_low', 's_adv_h_mid', 's_adv_h_high',
       'shear_low', 'shear_mid', 'shear_high',
       'omega_low', 'omega_mid', 'omega_high',
       'mse_low', 'mse_mid', 'mse_high',
       'prec_05']
y = 'prec_05'
data = summer[inputs].copy()
x = data.columns[data.columns !=y]
data['time'] = summer['time'].copy()
data['undiluteCAPE_zm_mu'] = summer['undiluteCAPE_zm_mu'].copy()
all = list(x)
all.append('time')
all.append('undiluteCAPE_zm_mu')

# normalize x data to mean = 0, std = 1
scaler = preprocessing.StandardScaler().fit(data[x])
x_scaled = scaler.transform(data[x])
data[x] = x_scaled

# split data into train and test (update later with validate)
split_index = round(len(data)/15 * 12)
train = data.iloc[0:split_index,:]#12 years
test = data.iloc[split_index::,:]#3 years

# reduce class inbalance by undersampling non-convection events
n0 = sum(data[y]==0.0) #calculating r based on whole dataset, should be approximate for train and test sets
n1 = sum(data[y]==1.0)
r = 3 # r = n0/n0' = original majority class size / resampled majority class size
sample_ratio = r*n1/n0
undersample = RandomUnderSampler(sampling_strategy=sample_ratio, random_state = SEED)
x_u_train, y_u_train = undersample.fit_resample(train[all], train[y]) #training set
x_u_test, y_u_test = undersample.fit_resample(test[all], test[y]) #test set

print("n0: ", n0)
print("n1: ", n1)
print("original ratio: ", round(n1/n0, 3))
print("undersampled ratio: ", round(sample_ratio,3))
print('r: ', r)

#%% random forest
rf = joblib.load('output/rf/rf.joblib')

#%% Model results
model_scores(rf, x, y, r, test, x_u_test[x], y_u_test, path, filename = '')

#%% Shaply Feature Importance
X100 = shap.utils.sample(x_u_test[x], 100) # 100 instances for use as the background distribution
X1000 = x_u_test[x] #shap.utils.sample(x_u_test[x], 1000)

#%%
# compute the SHAP values
# explainer = shap.Explainer(rf.predict, X1000)
# shap_values = explainer(X1000)

#%%

filename_expl = 'explainer.sav'
# pickle.dump(explainer, open(path+filename_expl, 'wb'))
explainer = pickle.load(open(path+filename_expl, 'rb'))

filename = 'shapvalues.sav'
# pickle.dump(shap_values, open(path+filename, 'wb'))
shap_values = pickle.load(open(path+filename, 'rb'))

s = shap_values.values
s.shape

#%% 
################ FEATURE IMPORTANCE FOR ALL THREE METHODS combined #############################
fig, ax = plt.subplots(1, 3, figsize=(7,8))
color1 = '#0072b2'
color2 = '#9d2f00'
######### IMPURITY IMPORTANCE ########
importances = rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf.estimators_],axis=0)
indices = np.argsort(importances)[::-1]

#ordered and named importances
var = list(x)
var_ordered = [var[i] for i in indices]
var_labels = [labels[i] for i in var_ordered]

ax[0].barh(range(x.shape[0]), importances[indices], color="lightseagreen", xerr=std[indices], align="center")
ax[0].set_yticks(range(x.shape[0]))  # Use set_yticks without ticks parameter
ax[0].set_yticklabels(var_labels, horizontalalignment='right')  # Use set_yticklabels
ax[0].set_ylim([x.shape[0], -1])
ax[0].set_xlim([-0.04, 0.5])
ax[0].set_xlabel("a. Impurity \n Feature Importance")
ax[0].xaxis.set_ticks_position('top')
ax[0].xaxis.set_label_position('top')
#color various top features
ax[0].get_yticklabels()[var_labels.index('ω mid')].set_color(color1)
ax[0].get_yticklabels()[var_labels.index('dilute dCAPE')].set_color(color1)
ax[0].get_yticklabels()[var_labels.index('ω high')].set_color(color1)
#ax[0].get_yticklabels()[var_labels.index('ω low')].set_color(color2)
ax[0].get_yticklabels()[var_labels.index('RH surface')].set_color(color2)
#ax[0].get_yticklabels()[var_labels.index('LCL')].set_color(color2)
ax[0].get_yticklabels()[var_labels.index('PW')].set_color(color2)
ax[0].get_yticklabels()[var_labels.index('T surface')].set_color(color2)
ax[0].get_yticklabels()[var_labels.index('T low')].set_color(color2)
#ax[0].get_yticklabels()[var_labels.index('q low')].set_color(color2)

######### PERMUTATION IMPORTANCE ############
perm = permutation_importance(rf, x_u_test[x], y_u_test, n_repeats=10, n_jobs = -1, random_state=0)                         
importances = perm.importances_mean
std = perm.importances_std
indices = np.argsort(importances)[::-1]

#ordered and named importances
var = list(x)
var_ordered = [var[i] for i in indices]
var_labels = [labels[i] for i in var_ordered]

ax[1].barh(range(x.shape[0]), importances[indices],
        color="lightseagreen", xerr=std[indices], align="center")
ax[1].set_yticks(range(x.shape[0]), var_labels,horizontalalignment='right',rotation_mode='anchor')
ax[1].set_ylim([x.shape[0],-1])
ax[1].set_xlabel("b. Permutation \n Feature Importance")
ax[1].xaxis.set_ticks_position('top')
ax[1].xaxis.set_label_position('top')
#color various top features
ax[1].get_yticklabels()[var_labels.index('ω mid')].set_color(color1)
ax[1].get_yticklabels()[var_labels.index('dilute dCAPE')].set_color(color1)
ax[1].get_yticklabels()[var_labels.index('ω high')].set_color(color1)
#ax[1].get_yticklabels()[var_labels.index('ω low')].set_color(color2)
ax[1].get_yticklabels()[var_labels.index('RH surface')].set_color(color2)
#ax[1].get_yticklabels()[var_labels.index('LCL')].set_color(color2)
ax[1].get_yticklabels()[var_labels.index('PW')].set_color(color2)
ax[1].get_yticklabels()[var_labels.index('T surface')].set_color(color2)
ax[1].get_yticklabels()[var_labels.index('T low')].set_color(color2)
#ax[1].get_yticklabels()[var_labels.index('q low')].set_color(color2)

########## SHAP FEATURE IMPORTANCE ########
importances = np.mean(abs(s), axis =0)
std = np.std(abs(s), axis =0)
indices = np.argsort(importances)[::-1]

#ordered and named importances
var = list(x)
var_ordered = [var[i] for i in indices]
var_labels = [labels[i] for i in var_ordered]

ax[2].barh(range(x.shape[0]), importances[indices],
        color="lightseagreen", xerr=std[indices], align="center")
ax[2].set_yticks(range(x.shape[0]), var_labels,horizontalalignment='right',rotation_mode='anchor')
ax[2].set_ylim([x.shape[0],-1])
ax[2].set_xlabel("c. SHAP Value \n Feature Importance")
ax[2].xaxis.set_ticks_position('top')
ax[2].xaxis.set_label_position('top')
#color various top features
ax[2].get_yticklabels()[var_labels.index('ω mid')].set_color(color1)
ax[2].get_yticklabels()[var_labels.index('dilute dCAPE')].set_color(color1)
ax[2].get_yticklabels()[var_labels.index('ω high')].set_color(color1)
#ax[2].get_yticklabels()[var_labels.index('ω low')].set_color(color2)
ax[2].get_yticklabels()[var_labels.index('RH surface')].set_color(color2)
#ax[2].get_yticklabels()[var_labels.index('LCL')].set_color(color2)
ax[2].get_yticklabels()[var_labels.index('PW')].set_color(color2)
ax[2].get_yticklabels()[var_labels.index('T surface')].set_color(color2)
ax[2].get_yticklabels()[var_labels.index('T low')].set_color(color2)
#ax[2].get_yticklabels()[var_labels.index('q low')].set_color(color2)

fig.tight_layout()
fig.savefig(path+'feature_importance_all_three_methods.png',dpi=500, facecolor='white', transparent=False)

#%%
######just SHAP feature importance for poster
fig, ax = plt.subplots(1, 1, figsize=(9,3))
importances = np.mean(abs(s), axis =0)
std = np.std(abs(s), axis =0)
indices = np.argsort(importances)[::-1]

#ordered and named importances
var = list(x)
var_ordered = [var[i] for i in indices]
var_labels = [labels[i] for i in var_ordered]

plt.bar(range(x.shape[0]), importances[indices],
        color="lightseagreen", yerr=std[indices], align="center")
plt.xticks(range(x.shape[0]), var_labels,rotation=45, horizontalalignment='right',rotation_mode='anchor')
plt.xlim([-1, x.shape[0]])
plt.ylabel("SHAP Value \n Feature Importance")

fig.tight_layout()
fig.savefig(path+'feature_importance_SHAP.png',dpi=500, facecolor='white', transparent=False)


#%% Elbow curve plot
X = s
distorsions = []
for k in range(2, 16):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    distorsions.append(kmeans.inertia_)

fig = plt.figure(figsize=(6, 3.5))
plt.plot(range(2, 16), distorsions, marker = 'o')
plt.grid(True)
#plt.title('Elbow curve')
plt.xlabel('Number of clusters')
plt.ylabel('Within-cluster variance')#sum of squared distances
plt.tight_layout()
plt.xticks([2,3,4,5,6,7,8,9,10,11,12,13,14,15])
plt.savefig(path+'elbow_plot_k_means.png',dpi=800, facecolor='white', transparent=False)
     

#%% K means clustering
kmeans = KMeans(n_clusters=nclusters, random_state=SEED).fit(X)
kmeans.labels_

colors = ["#9d2f00",
"#e69f00",
"#f0e36e",
"#009e73",
"#56b4e9",
"#0072b2"]
#markers = ['o','o','s','s','o',]
#colors = ['#f5b041','#c8fa72','#7D3C98', '#e74c3c', '#16a085','#2e86c1','#FFFF00', '#B2EBF2' ]
colors = colors[0:nclusters]
cmap = mpl.colors.ListedColormap(colors)

clusters = X1000.copy()
clusters['proba'] = rf.predict_proba(X1000[x])[:, 1]
clusters['proba_Milo'] = clusters['proba']/(r + (1-r)*clusters['proba'])
clusters['prec_05'] = y_u_test
clusters['cluster'] = kmeans.labels_
#re-map the clusters so they're in an order that makes sense
map= {2:0, 1:1, 5:2, 0:3, 4:4, 3:5}
clusters['cluster_map'] = np.vectorize(map.get)(clusters['cluster'])

clusters_unscaled = pd.DataFrame(data = scaler.inverse_transform(X1000), columns = x)
clusters_unscaled['cluster'] = kmeans.labels_
clusters_unscaled['prec_05'] = clusters['prec_05']
clusters_unscaled['proba'] = clusters['proba']
clusters_unscaled['cluster_map'] = np.vectorize(map.get)(clusters_unscaled['cluster'])

clusters_diurnal = clusters_unscaled.copy()
clusters_diurnal['time'] = x_u_test['time']
clusters_diurnal['proba_Milo'] = clusters_diurnal['proba']/(r + (1-r)*clusters_diurnal['proba'])

shaps = pd.DataFrame(s)
shaps.columns = X1000.columns
shaps.index = X1000.index
shaps['cluster'] = kmeans.labels_
shaps['cluster_map'] = np.vectorize(map.get)(shaps['cluster'])

#%% Histograms of convection probability and occurrence
hists = clusters.groupby(['cluster_map'])
nrows = int(np.ceil(nclusters/2))
ncols = 2

fig = plt.figure(figsize=(5.5,11))
for i, group in zip (range(nclusters), hists):
    ax = plt.subplot(nclusters, 2, i*2+1)
    plt.hist(group[1]['proba'], bins = np.linspace(0,1,21), color = colors[i])
    plt.title('Cluster '+str(i+1), size = 11)
    if i == (nclusters-1):
        plt.xlabel('Predicted probability \n of convection', size = 11)
    else: plt.xlabel('')
    ax = plt.subplot(nclusters, 2, i*2+2)
    plt.hist(group[1]['prec_05'], bins = np.linspace(0,1,21), color = colors[i])
    plt.title('Cluster '+str(i+1), size = 11)
    if i == (nclusters-1):
        plt.xlabel('Observed occurrence \n of convection', size = 11)
    else: plt.xlabel('')

#fig.text(0.27, -0.006, 'Predicted probability \n of convection', size=12, ha='center', va='bottom')
#fig.text(0.76, -0.006, 'Observed occurrence of convection', size=12, ha='center', va='bottom')
fig.tight_layout()
fig.savefig(path+'shaply_clusters_histograms_probabilities_observations.png', bbox_inches='tight', dpi=800, facecolor='white', transparent=False)


#%%Combine average characteristics into 1 plot
#Average feature characteristics of each cluster
groups = clusters.groupby(['cluster_map']).mean()
df = np.array(groups)
var_labels = [labels[i] for i in shaps.columns[0:len(x)]]

fig = plt.figure(figsize=(8,11))
for i in range(nclusters):
    ax = plt.subplot(2, nclusters, i+1)
    plt.axvline(x=0, color='gray', linestyle='-', alpha = 0.5, linewidth = 1)
    plt.barh(clusters.columns[0:len(x)], df[i,0:len(x)], color = colors[i])
    if i > 0:
        plt.tick_params(labelleft = False, left = False)
    if i == 0:
        plt.yticks(list(range(len(x))), var_labels)#, rotation = 45, ha='right',rotation_mode='anchor')
    plt.ylim([-1,len(x)])
    plt.gca().invert_yaxis()
    plt.title('Cluster '+str(i+1), size=10)
    plt.grid(axis = 'y', alpha = 0.8)
    plt.xticks(rotation = 45, ha='right',rotation_mode='anchor')

#Average shap value characteristics of each cluster
groups = shaps.groupby(['cluster_map']).mean()
df = np.array(groups)
for i in range(nclusters):
    ax = plt.subplot(2, nclusters, i+1+nclusters)
    plt.axvline(x=0, color='gray', linestyle='-', alpha = 0.5, linewidth = 1)
    plt.barh(shaps.columns[0:len(x)], df[i,0:len(x)], color = colors[i])
    if i > 0:
        plt.tick_params(labelleft = False, left = False)
    if i == 0:
        plt.yticks(list(range(len(x))), var_labels)#, rotation = 45, ha='right',rotation_mode='anchor')
    plt.ylim([-1,len(x)])
    plt.gca().invert_yaxis()
    plt.grid(axis = 'y', alpha = 0.8)
    plt.title('    ', size = 20)
    plt.xticks(rotation = 45, ha='right',rotation_mode='anchor')

fig.text(0.56, 0.52, 'Feature values (scaled)', size=11, ha='center', va='center')
fig.text(0.56, 0.007, 'SHAP values', size=11, ha='center', va='center')
fig.text(0.05, 0.99, 'a.', size=13, ha='center', va='center')
fig.text(0.05, 0.485, 'b.', size=13, ha='center', va='center')
fig.tight_layout()
plt.subplots_adjust(wspace=0.15)
fig.savefig(path+'shaply_clusters_histograms_features_AND_shapvalues_paper.png',dpi=800, facecolor='white', transparent=False)


#%%Combine average characteristics into 1 plot FOR POSTER
#Average feature characteristics of each cluster

fig = plt.figure(figsize=(13,14))
groups = clusters.groupby(['cluster_map']).mean()
df = np.array(groups)
var_labels = [labels[i] for i in shaps.columns[0:len(x)]]

for i in range(nclusters):
    ax = plt.subplot(nclusters, 2, i*2+1)
    plt.axhline(y=0, color='gray', linestyle='-', alpha = 0.5, linewidth = 1)
    plt.bar(clusters.columns[0:len(x)], df[i,0:len(x)], color = colors[i])
    if i < (nclusters-1):
        plt.tick_params(labelbottom = False, bottom = False)
    if i == (nclusters-1):
        plt.xticks(list(range(len(x))), var_labels, rotation = 45, ha='right',rotation_mode='anchor')
    plt.title('    ')
    plt.grid(axis = 'x', alpha = 0.8)
#Average shap value characteristics of each cluster
groups = shaps.groupby(['cluster_map']).mean()
df = np.array(groups)
for i in range(nclusters):
    ax = plt.subplot(nclusters, 2, i*2+2)
    plt.axhline(y=0, color='gray', linestyle='-', alpha = 0.5, linewidth = 1)
    plt.bar(shaps.columns[0:len(x)], df[i,0:len(x)], color = colors[i])
    if i < (nclusters-1):
        plt.tick_params(labelbottom = False, bottom = False)
    if i == (nclusters-1):
        plt.xticks(list(range(len(x))), var_labels, rotation = 45, ha='right',rotation_mode='anchor')
    plt.title('    ')
    plt.grid(axis = 'x', alpha = 0.8)
fig.text(0.275, 0.97, 'Feature values (scaled)', size=11, ha='center', va='center')
fig.text(0.775, 0.97, 'SHAP values', size=11, ha='center', va='center')
fig.tight_layout()
fig.savefig(path+'shaply_clusters_histograms_features_AND_shapvalues_poster.png',dpi=800, facecolor='white', transparent=False)


#%%Combine average characteristics into 1 plot, with SAME XAXIS
#Average feature characteristics of each cluster
groups = clusters.groupby(['cluster_map']).mean()
df = np.array(groups)
var_labels = [labels[i] for i in shaps.columns[0:len(x)]]

fig = plt.figure(figsize=(8,11))
for i in range(nclusters):
    ax = plt.subplot(2, nclusters, i+1)
    plt.axvline(x=0, color='gray', linestyle='-', alpha = 0.5, linewidth = 1)
    plt.barh(clusters.columns[0:len(x)], df[i,0:len(x)], color = colors[i])
    if i > 0:
        plt.tick_params(labelleft = False, left = False)
    if i == 0:
        plt.yticks(list(range(len(x))), var_labels)#, rotation = 45, ha='right',rotation_mode='anchor')
    plt.ylim([-1,len(x)])
    plt.xlim([-3,3])
    plt.gca().invert_yaxis()
    plt.title('Cluster '+str(i+1), size=10)
    plt.grid(axis = 'y', alpha = 0.8)
    plt.xticks(rotation = 45, ha='right',rotation_mode='anchor')

#Average shap value characteristics of each cluster
groups = shaps.groupby(['cluster_map']).mean()
df = np.array(groups)
for i in range(nclusters):
    ax = plt.subplot(2, nclusters, i+1+nclusters)
    plt.axvline(x=0, color='gray', linestyle='-', alpha = 0.5, linewidth = 1)
    plt.barh(shaps.columns[0:len(x)], df[i,0:len(x)], color = colors[i])
    if i > 0:
        plt.tick_params(labelleft = False, left = False)
    if i == 0:
        plt.yticks(list(range(len(x))), var_labels)#, rotation = 45, ha='right',rotation_mode='anchor')
    plt.ylim([-1,len(x)])
    plt.xlim([-.2,.2])
    plt.gca().invert_yaxis()
    plt.grid(axis = 'y', alpha = 0.8)
    plt.title('    ', size = 20)
    plt.xticks(rotation = 45, ha='right',rotation_mode='anchor')

fig.text(0.56, 0.52, 'Feature values (scaled)', size=11, ha='center', va='center')
fig.text(0.56, 0.007, 'SHAP values', size=11, ha='center', va='center')
fig.text(0.05, 0.99, 'a.', size=13, ha='center', va='center')
fig.text(0.05, 0.485, 'b.', size=13, ha='center', va='center')
fig.tight_layout()
plt.subplots_adjust(wspace=0.15)
fig.savefig(path+'shaply_clusters_histograms_features_AND_shapvalues_poster_FIX_X_AXIS.png',dpi=800, facecolor='white', transparent=False)


#%% Scatters of clusters in real feature space and shap value space
units = ['hPa hr$^{-1}$','J kg$^{-1}$ hr$^{-1}$','-', 'mm']
pt = plt.figure(figsize=(9,6))
plt.subplot(231)
plt.scatter(shaps['omega_mid'], shaps['PW_q'], s = 14,c = shaps['cluster_map'], cmap = cmap)
plt.xlabel(labels['omega_mid'])
plt.ylabel(labels['PW_q'])
plt.title(' ', size = 14)
plt.subplot(232)
scat = plt.scatter(shaps['dCAPE_zm_dilute_tzhang'], shaps['LCL'], s = 14,c = shaps['cluster_map'], cmap = cmap)
plt.xlabel(labels['dCAPE_zm_dilute_tzhang'])
plt.ylabel(labels['LCL'])
plt.title(' ', size = 14)
plt.subplot(233)
plt.scatter(shaps['rh'], shaps['T_srf'], s = 14,c = shaps['cluster_map'], cmap = cmap)
plt.xlabel(labels['rh'])
plt.ylabel(labels['T_srf'])
plt.title(' ', size = 14)
plt.subplot(234)
plt.scatter(clusters_unscaled['omega_mid'], clusters_unscaled['PW_q'], s = 14,c = clusters['cluster_map'], cmap = cmap)
plt.xlabel(labels['omega_mid']+ ' ('+units[0]+')')
plt.ylabel(labels['PW_q']+ ' ('+units[3]+')')
plt.title(' ', size = 14)
#plt.legend(*scat.legend_elements(), title="Clusters",frameon=False, ncol = 2, prop={'size': 8})
cluster_labels = ['1', '2', '3', '4', '5', '6']
legend_labels = [Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=color, markersize=8) for label, color in zip(cluster_labels, cmap.colors)]
plt.legend(handles=legend_labels, title="Clusters", frameon=False, ncol=2, prop={'size': 8})
plt.subplot(235)
plt.scatter(clusters_unscaled['dCAPE_zm_dilute_tzhang'], clusters_unscaled['LCL'], s = 14, c = clusters['cluster_map'], cmap = cmap)
plt.xlabel(labels['dCAPE_zm_dilute_tzhang']+ ' ('+units[1]+')')
plt.ylabel(labels['LCL']+ ' (hPa)')
plt.title(' ', size = 14)
plt.subplot(236)
scat = plt.scatter(clusters_unscaled['rh'], clusters_unscaled['T_srf'], s = 14, c = clusters['cluster_map'], cmap = cmap)
plt.xlabel(labels['rh']+ ' ('+units[2]+')')
plt.ylabel(labels['T_srf']+ ' (°C)')
plt.title(' ', size = 14)
pt.suptitle(' ',size = 14)
y = 0.92
pt.text(0.02, 0.91, 'a.',size = 13)
pt.text(0.54, 0.90, 'SHAP values',size = 11, ha='center', va='center')
pt.text(0.02, 0.46, 'b.',size = 13)
pt.text(0.54, 0.45, 'Real feature values',size = 11, ha='center', va='center')
pt.tight_layout()
pt.savefig(path+'shaply_clusters_top_features_scatter_BOTH.png',dpi=800, facecolor='white', transparent=False)


#%% Partial dependence plot
groups_features = clusters.groupby(['cluster_map'])
groups_features_unscaled = clusters_unscaled.groupby(['cluster_map'])
groups_shaps = shaps.groupby(['cluster_map'])

names = [ 'omega_mid', 'dCAPE_zm_dilute_tzhang', 'LCL', 'rh']
units = ['hPa hr$^{-1}$','J kg$^{-1}$hr$^{-1}$', 'hPa','-']
i = 0
plt.figure(figsize = (7.1,6))
for name in names:
    plt.subplot(2,2,i+1)
    for group_feature_un, group_shap in zip(groups_features_unscaled, groups_shaps):
        x = group_feature_un[1][name]
        y = group_shap[1][name]
        xy = pd.concat([group_feature_un[1][name],group_shap[1][name]], axis = 1)
        xy.columns = ['x', 'y']
        xy = xy.sort_values(by = 'x')
        smoothed = gaussian_filter1d(xy['y'], sigma = 17)
        plt.scatter(group_feature_un[1][name], group_shap[1][name], s = 14,color = colors[group_feature_un[0]],edgecolors='None',alpha = 0.2)
        plt.plot(xy['x'], smoothed, color = colors[group_feature_un[0]], linewidth = 3, alpha = 0.95, label = str(group_feature_un[0]+1))
        if name == names[0]:
            plt.legend(title="Clusters",frameon=False, ncol =2)
        plt.xlabel('Feature value for '+labels[name]+ ' ('+units[i]+')')
        plt.ylabel('SHAP value for '+labels[name])
    i=i+1
plt.tight_layout()
plt.savefig(path+'partial_dependence_plot_smoothed_unscaled_4_panel.png',dpi=800, facecolor='white', transparent=False)
             

#%% Diurnal cycle of clusters
d = clusters_diurnal.copy()
d['timef'] = pd.to_datetime(d['time'])
d['hour'] = [hour_rounder(t) for t in d['timef']]
d['count'] = np.ones(len(d))

#fix time to local time (SGP is 5 hours behind UTC)
d['hour_lst'] = d['hour'].dt.hour -5
d['hour_lst'][d['hour_lst'] < 0] += 24

fig,ax = plt.subplots(figsize=(11,7))
for i in range(nclusters):
    f = d[d['cluster_map']==i]
    times = f['hour_lst']
    hourly_sum = f.groupby(times).sum()
    hourly_mean = f.groupby(times).mean()
    ax.plot(hourly_mean.index, hourly_mean['proba_Milo'], '--', label = 'Cluster '+str(i+1),color=colors[i],marker="^")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Probability of Convection")
    ax.legend(frameon = False, loc = 'upper left')
fig.savefig(path+'diurnal_cycle_proba_Milo.png',dpi=500, facecolor='white', transparent=False)

fig,ax = plt.subplots(figsize=(12,8))
for i in range(nclusters):
    f = d[d['cluster_map']==i]
    times = f['hour_lst']
    hourly_sum = f.groupby(times).sum()
    hourly_mean = f.groupby(times).mean()
    ax.plot(hourly_mean.index, hourly_mean['prec_05'], '--', label = 'Cluster '+str(i+1),color=colors[i],marker="^")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Occurrence of Convection")
    ax.legend(frameon = False)
fig.savefig(path+'diurnal_cycle_observed.png',dpi=500, facecolor='white', transparent=False)

#count of each cluster
fig,ax = plt.subplots(figsize=(12,8))
for i in range(nclusters):
    f = d[d['cluster_map']==i]
    times = f['hour_lst']
    hourly_sum = f.groupby(times).sum()
    hourly_mean = f.groupby(times).mean()
    ax.plot(hourly_sum.index, hourly_sum['count'], '--', label = 'Cluster '+str(i+1),color=colors[i],marker="^")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Count")
    ax.legend(frameon = False)
fig.savefig(path+'diurnal_cycle_count.png',dpi=500, facecolor='white', transparent=False)

#%%


# %%
