"""

Compare ML trigger to CAPE-based thresholds

"""
#%%
from rf_functions import *
import joblib
from matplotlib.gridspec import GridSpec

#%% ETS score function

def ETS(true, predicted):
    a = np.sum((true==1)&(predicted==1)) # true positives
    b = np.sum((true==0)&(predicted==1)) # overprediction
    c = np.sum((true==1)&(predicted==0)) # underprediction
    d = np.sum((true==0)&(predicted==0)) # true negatives
    n = a+b+c+d
    ets = (a-((a+b)*(a+c))/n)/(a+b+c-((a+b)*(a+c))/n)
    return ets


#%%
path = 'output/cape_thresholds_s_adv/'
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


#%% Create random forest
rf = joblib.load('output/rf/rf.joblib')


# %% Model results
#test dataset
rf_scores = model_scores(rf, x, y, r, test, x_u_test[x], y_u_test, path, filename = '')


#%%
'''Miloshevich probability post-ML adjustment'''
proba_full = rf.predict_proba(test[x])[:, 1]
proba_updated = proba_full/(r + (1-r)*proba_full)
color1 = '#1f77b4'
color2 = '#8FBD70'


#%%
#Reliability curve
pt = plt.figure(figsize=(4,4))
plt.plot([0, 1], [0, 1], "k--",label="Perfectly calibrated")
plt.ylabel("Observed frequency in bin")
plt.xlabel("Predicted probability in bin")
fraction_of_positives_rfc, mean_predicted_value_rfc = calibration_curve(test[y], proba_full, n_bins=20)
plt.plot(mean_predicted_value_rfc, fraction_of_positives_rfc, "s-", label="%s" % ('Model, unadjusted'))
fraction_of_positives_rfc, mean_predicted_value_rfc = calibration_curve(test[y], proba_updated, n_bins=20)
plt.plot(mean_predicted_value_rfc, fraction_of_positives_rfc, "s-", label="%s" % ('Model, adjusted'))
plt.legend(loc="lower right", frameon=False)
pt.tight_layout()
pt.savefig(path+'reliability_curve_balanced_Miloshevich_update.png',dpi=500, facecolor='white', transparent=False)


#Diurnal cycle
d = test.copy()
d['proba'] = proba_full
d['proba_Milo'] = proba_updated
d['predict'] = rf.predict(test[x])
d['predict_Milo'] = np.multiply((proba_updated>0.5),1.0)
d['timef'] = pd.to_datetime(d['time'])
d['hour'] = [hour_rounder(t) for t in d['timef']]
times = d['hour'].dt.hour
hourly_sum = d.groupby(times).sum()
hourly_mean = d.groupby(times).mean()
fig,ax = plt.subplots(figsize=(12,8))
ax.plot(hourly_mean.index, hourly_mean[y], label = 'Observed',marker=".", color = 'black')
ax.plot(hourly_mean.index, hourly_mean['predict'], label = 'Random forest count, unadjusted',marker=".", color = '#1f77b4')
ax.plot(hourly_mean.index, hourly_mean['predict_Milo'], label = 'Random forest count, adjusted',marker=".", color = 'orange')
ax.set_ylim([0,0.3])
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Number of Convective Events / Total Length of Dataset", color = 'black')
#ax.legend(frameon=False)
ax2=ax.twinx()
ax2.plot(hourly_mean.index, hourly_mean['proba'], '--', label = 'Random forest probability, unadjusted',color='#1f77b4',marker="^")
ax2.plot(hourly_mean.index, hourly_mean['proba_Milo'], '--', label = 'Random forest probability, adjusted',color='orange',marker="^")
ax2.set_ylabel("Probability of Convection",color='black')
ax2.set_ylim([0,0.3])
fig.legend(frameon=False, loc='upper right', bbox_to_anchor=(0.85, 0.93))
fig.tight_layout()
fig.savefig(path+'diurnal_cycle_proba_Milo.png',dpi=500, facecolor='white', transparent=False)


#%% CAPE-based thresolds
# Find optimal thresholds for this dataset, optimize F1
capes = ['undiluteCAPE_zm_mu', 'diluteCAPE_zm_mu', 'dCAPE_zm_undilute_tzhang', 'dCAPE_zm_dilute_tzhang']
train_summer = summer.iloc[0:split_index,:]
test_summer = summer.iloc[split_index::,:]
thresh = np.zeros(4)*np.nan
thresh_brier = np.zeros(4)*np.nan
thresh_ets = np.zeros(4)*np.nan
f1 = np.zeros(4)*np.nan
brier = np.zeros(4)*np.nan
ets = np.zeros(4)*np.nan
j=0
for cape in capes:
    # train on unbalanced original training dataset
    true_labels = train_summer.prec_05 #(0 for non-convective, 1 for convective)
    cape_values = train_summer[cape]
    increment = int((np.max(cape_values)-np.min(cape_values))/80)###change back to more
    thresholds = np.arange(int(np.min(cape_values)), int(np.max(cape_values)), 1)
    tp = np.zeros(len(thresholds))*np.nan
    fp = np.zeros(len(thresholds))*np.nan
    fn = np.zeros(len(thresholds))*np.nan
    f1_scores = np.zeros(len(thresholds))*np.nan
    p_scores = np.zeros(len(thresholds))*np.nan
    r_scores = np.zeros(len(thresholds))*np.nan
    brier_scores = np.zeros(len(thresholds))*np.nan
    ets_scores = np.zeros(len(thresholds))*np.nan

    # Iterate through different threshold values
    for i in range(len(thresholds)):
        threshold = thresholds[i]
        predicted_labels = np.where(cape_values >= threshold, 1, 0)
        tp[i] = np.sum((true_labels==1) & (predicted_labels==1))
        fp[i] = np.sum((true_labels==0) & (predicted_labels==1))
        fn[i] = np.sum((true_labels==1) & (predicted_labels==0))
        p_scores[i] = precision_score(true_labels, predicted_labels)
        r_scores[i] = recall_score(true_labels, predicted_labels)
        f1_scores[i] = f1_score(true_labels, predicted_labels)
        brier_scores[i] = brier_score_loss(true_labels, predicted_labels)
        ets_scores[i] = ETS(true_labels, predicted_labels)

    print(cape)
    print('Best F1 Score:'+str(np.max(f1_scores)))
    print('Optimal Threshold:'+str(thresholds[f1_scores==np.max(f1_scores)]))
    print('---------------------------')

    thresh[j] = thresholds[f1_scores==np.max(f1_scores)]
    f1[j] = np.max(f1_scores)
    thresh_brier[j] = thresholds[brier_scores==np.min(brier_scores)][0]
    brier[j] = np.min(brier_scores)
    thresh_ets[j] = thresholds[ets_scores==np.max(ets_scores)][0]
    ets[j] = np.max(ets_scores)
    j = j+1

    fig, ax1 = plt.subplots()
    # Plot F1 score on the left y-axis
    ax1.plot(thresholds, f1_scores, 'b-', label='F1 Score')
    ax1.plot(thresholds, ets_scores, 'c-', label='ETS')
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('F1 Score', color='b')
    ax1.tick_params('y', colors='b')
    # Create a twin axis for precision and recall
    ax2 = ax1.twinx()
    # Plot precision and recall on the right y-axis
    ax2.plot(thresholds, p_scores, 'r-', label='Precision')
    ax2.plot(thresholds, r_scores, 'g-', label='Recall')
    ax2.plot(thresholds, brier_scores, 'y-', label='Brier Score')
    ax2.set_ylabel('Precision and Recall', color='r')
    ax2.tick_params('y', colors='r')
    # Add legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='best', frameon = False)
    plt.title(cape)
    fig.tight_layout()
    fig.savefig(path+'threshold_optimize_'+cape+'.png',dpi=500, facecolor='white', transparent=False)

    fig, ax = plt.subplots()
    ax.plot(thresholds, tp, 'r-', label='tp')
    ax.plot(thresholds, fn, 'b-', label='fn')
    ax.plot(thresholds, fp, 'g-', label='fp')
    ax.set_xlabel('Threshold')
    ax.legend(frameon = False)
    fig.tight_layout()
    fig.savefig(path+'threshold_optimize_tp_fp_fn_'+cape+'.png',dpi=500, facecolor='white', transparent=False)

#%%
#evaluate scores on the test set instead of the train set
f1_test = np.zeros(4)*np.nan
brier_test = np.zeros(4)*np.nan
ets_test = np.zeros(4)*np.nan
p_test = np.zeros(4)*np.nan
r_test = np.zeros(4)*np.nan
f1_tzhang = np.zeros(4)*np.nan
thresh_tzhang = [70,70,65,65]
#optimized on F1, other scores are for the threshold where F1 is optimized
for i in range(4):
    true_labels = test_summer.prec_05 #(0 for non-convective, 1 for convective)
    cape_values = test_summer[capes[i]]
    # F1 for test set using optimized thresholds
    predicted_labels = np.where(cape_values >= thresh[i], 1, 0)
    f1_test[i] = f1_score(true_labels, predicted_labels)
    p_test[i] =  precision_score(true_labels, predicted_labels)
    r_test[i] =  recall_score(true_labels, predicted_labels)
    brier_test[i] = brier_score_loss(true_labels, predicted_labels)
    ets_test[i] = ETS(true_labels, predicted_labels)
    #F1 for test set using Zhang threshold
    predicted_labels = np.where(cape_values >= thresh_tzhang[i], 1, 0)
    f1_tzhang[i] = f1_score(true_labels, predicted_labels)


#put all scores into a csv file
scores = {'Index': capes,
          'Threshold F1': thresh,
          'Threshold Brier': thresh_brier,
          'Threshold ETS': thresh_ets,
          'Zhang thresholds': thresh_tzhang,
          'F1, train': f1,
          'F1, test': f1_test,
          'P, test': p_test,
          'R, test': r_test,
          'Brier, train': brier, 
          'Brier, test': brier_test, 
          'ETS, train': ets, 
          'ETS, test': ets_test, 
          'F1, test tzhang': f1_tzhang,
          }
df = pd.DataFrame(data=scores)
df.to_csv(path+'thresholds.csv') 

#%% bar plot to compare scores

s = {'Index': ['undiluteCAPE_zm_mu', 'diluteCAPE_zm_mu', 'dCAPE_zm_undilute_tzhang', 'dCAPE_zm_dilute_tzhang','rf'],
    'F1, test': np.append(f1_test, rf_scores['F1'][1]),
    'P, test': np.append(p_test,rf_scores['Precision'][1]),
    'R, test': np.append(r_test,rf_scores['Recall'][1]),
    'Brier, test': np.append(brier_test,rf_scores['Brier'][1])
          }

#Caluclate Brier skill score
climate = (n1/n0)*np.ones(len(test_summer)) #just average frequency of rain
brier_ref = brier_score_loss(true_labels, climate)
s['BSS, test'] = 1 - s['Brier, test']/brier_ref
diurnal = np.tile(hourly_mean[y].values, int(len(test_summer)/24)) #hourly diurnal frequency of rain
brier_ref_diurnal = brier_score_loss(true_labels, diurnal)
s['BSS diurnal, test'] = 1 - s['Brier, test']/brier_ref_diurnal

#BSS if rf trigger is deterministic
#test dataset: 'test' is the same as 'test_summer', but 'test' has normailzed features
model_scores(rf, x, y, r, test, x_u_test[x], y_u_test, path, filename = '')
predict_y = rf.predict(test[x])
proba = rf.predict_proba(test[x])[:, 1]
proba_adjust = proba/(r + (1-r)*proba)
predict_y_Milo = np.round(proba_adjust)
brier_det = brier_score_loss(test[y], predict_y_Milo)
brier_prob = brier_score_loss(test[y], proba_adjust)
BSS_det = 1 - brier_det/brier_ref_diurnal
BSS_prob = 1 - brier_prob/brier_ref_diurnal
print('BSS deterministic: ', round(BSS_det,3))
print('BSS probabilistic: ', round(BSS_prob,3))

#undersampled vs original (scaled) dataset
balanced = rf_scores['Brier'][2]
print('BSS, undersampled: ', str(1 - balanced/brier_ref_diurnal))
scaled = rf_scores['Brier'][1]
print('BSS, scaled original: ', str(1 - scaled/brier_ref_diurnal))

df = pd.DataFrame(data=s)
df.to_csv(path+'thresholds_more_scores.csv') 

# Extracting relevant data for plotting
metrics = ['P, test', 'R, test', 'F1, test', 'BSS diurnal, test']
metric_names = ['Precision', 'Recall', 'F1', 'Brier Skill Score']
index_labels = ['CAPE','Dilute CAPE','dCAPE','Dilute dCAPE', 'Random forest']
colors = ['#884EA0', '#FF8A65', '#F1C40F', '#E74C3C', color2]
data = {metric: s[metric] for metric in metrics}

# Bar width for each bar
bar_width = 0.13

# Bar positions for each index cluster
index_positions = np.arange(len(metrics))

fig, ax = plt.subplots(figsize=(6.5, 3))
# Plotting bars for each index and cluster
for i, index in enumerate(index_labels):
    ax.bar(index_positions + i*bar_width, [data[metric][i] for metric in metrics], width=bar_width, color = colors[i], label=index,align='center', alpha = 0.9)
ax.axhline(linewidth=0.5, color='gray')
ax.text(2.96, -0.01, str(round(s['BSS diurnal, test'][0],1)), size = 8, rotation = 90, verticalalignment='top')
ax.text(3.09, -0.01, str(round(s['BSS diurnal, test'][1],1)), size = 8, rotation = 90, verticalalignment='top')
# Setting labels and title
ax.set_ylabel('Score')
ax.set_xticks(index_positions+bar_width*2)
ax.set_xticklabels(metric_names)
ax.legend(frameon = False, ncol= 2)
ax.set_ylim([-0.18,1.07])

fig.tight_layout()
fig.savefig(path+'scores_bar_chart.png',dpi=500, facecolor='white', transparent=False)


# %%
#Diurnal cycle for optimized thresholds
d = test.copy()
d['proba'] = proba_full
d['proba_Milo'] = proba_updated
d['predict'] = rf.predict(test[x])
d['predict_Milo'] = np.multiply((proba_updated>0.5),1.0)
d['timef'] = pd.to_datetime(d['time'])
d['hour'] = [hour_rounder(t) for t in d['timef']]
d['CAPE'] = np.where(test_summer[capes[0]] >= thresh[0], 1, 0)
d['CAPE_dilute'] = np.where(test_summer[capes[1]] >= thresh[1], 1, 0)
d['dCAPE'] = np.where(test_summer[capes[2]] >= thresh[2], 1, 0)
d['dCAPE_dilute'] = np.where(test_summer[capes[3]] >= thresh[3], 1, 0)
times = d['hour'].dt.hour
hourly_sum = d.groupby(times).sum()
hourly_mean = d.groupby(times).mean()


#fix time to local time (SGP is 5 hours behind UTC)
lst_time = np.arange(24) - 5
lst_time[lst_time < 0] += 24
hourly_mean['hour_lst'] = lst_time
hourly_mean = hourly_mean.sort_values('hour_lst')


#%% for poster
########plotting
fig, ax = plt.subplots(figsize=(7,6))
ax.plot(hourly_mean['hour_lst'], hourly_mean[y], label = 'Observed frequency',marker=".", color = 'black')
ax.plot(hourly_mean['hour_lst'], hourly_mean['proba_Milo'], '--', label = 'Model probability',color=color2,marker="^")
ax.plot(hourly_mean['hour_lst'], hourly_mean['CAPE'], label = 'CAPE',marker="x", color = '#884EA0')
ax.plot(hourly_mean['hour_lst'], hourly_mean['CAPE_dilute'], label = 'dilute CAPE',marker="x", color = '#FF8A65')
ax.plot(hourly_mean['hour_lst'], hourly_mean['dCAPE'], label = 'dCAPE',marker="x", color = '#F1C40F')
ax.plot(hourly_mean['hour_lst'], hourly_mean['dCAPE_dilute'], label = ' dilute dCAPE',marker="x", color = '#E74C3C')
ax.set_ylim([0,0.9])
ax.set_ylabel("Probability or frequency of convection", color = 'black')
ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(0.05, 0.85), ncol = 3)



#%%
fig= plt.figure(figsize=(7,6))
gs = GridSpec(nrows=2, ncols=1, height_ratios=[3, 2])
gs.update(wspace=0.025, hspace=0.15)

ax = fig.add_subplot(gs[0, 0])
ax.plot(hourly_mean.hour_lst, hourly_mean[y], label = 'Observed frequency',marker=".", color = 'black')
ax.plot(hourly_mean.hour_lst, hourly_mean['proba_Milo'], '--', label = 'Model probability',color=color2,marker="^")
ax.plot(hourly_mean.hour_lst, hourly_mean['CAPE'], label = 'CAPE',marker=".", color = '#884EA0')
ax.plot(hourly_mean.hour_lst, hourly_mean['CAPE_dilute'], label = 'Dilute CAPE',marker=".", color = '#FF8A65')
ax.plot(hourly_mean.hour_lst, hourly_mean['dCAPE'], label = 'dCAPE',marker=".", color = '#F1C40F')
ax.plot(hourly_mean.hour_lst, hourly_mean['dCAPE_dilute'], label = ' Dilute dCAPE',marker=".", color = '#E74C3C')
ax.set_ylim([-0.1,0.9])
#ax.set_ylabel("Probability or frequency of convection", color = 'black')
ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(0.05, 0.89), ncol = 3)

#same figure, but zoomed in on specific features
ax1 = fig.add_subplot(gs[1, 0])
ax1.plot(hourly_mean.hour_lst, hourly_mean[y], label = 'Observed',marker=".", color = 'black')
ax1.plot(hourly_mean.hour_lst, hourly_mean['dCAPE'], label = 'dCAPE',marker=".", color = '#F1C40F')
ax1.plot(hourly_mean.hour_lst, hourly_mean['dCAPE_dilute'], label = 'Dilute dCAPE',marker=".", color = '#E74C3C')
ax1.plot(hourly_mean.hour_lst, hourly_mean['proba_Milo'], '--', label = 'Model probability, adjusted',color=color2,marker="^")
ax1.set_ylim([0,0.17])
ax1.set_xlabel("Hour of Day")
plt.ylabel("Probability or frequency of convection", y=1.4) 
fig.savefig(path+'diurnal_cycle_proba_Milo_CAPE_thresholds_optimized_double_panel.png',dpi=500, facecolor='white', transparent=False)




# %%
#Diurnal cycle for T Zhang paper thresholds
d = test.copy()
d['proba'] = proba_full
d['proba_Milo'] = proba_updated
d['predict'] = rf.predict(test[x])
d['predict_Milo'] = np.multiply((proba_updated>0.5),1.0)
d['timef'] = pd.to_datetime(d['time'])
d['hour'] = [hour_rounder(t) for t in d['timef']]
d['CAPE'] = np.where(test_summer[capes[0]] >= thresh_tzhang[0], 1, 0)
d['CAPE_dilute'] = np.where(test_summer[capes[1]] >= thresh_tzhang[1], 1, 0)
d['dCAPE'] = np.where(test_summer[capes[2]] >= thresh_tzhang[2], 1, 0)
d['dCAPE_dilute'] = np.where(test_summer[capes[3]] >= thresh_tzhang[3], 1, 0)
times = d['hour'].dt.hour
hourly_sum = d.groupby(times).sum()
hourly_mean = d.groupby(times).mean()
fig,ax = plt.subplots(figsize=(12,8))
ax.plot(hourly_mean.index, hourly_mean[y], label = 'Observed',marker=".", color = 'black')
ax.plot(hourly_mean.index, hourly_mean['predict'], label = 'Random forest count, unadjusted',marker=".", color = '#1f77b4')
ax.plot(hourly_mean.index, hourly_mean['predict_Milo'], label = 'Random forest count, adjusted',marker=".", color = 'orange')
ax.plot(hourly_mean.index, hourly_mean['CAPE'], label = 'CAPE',marker="x", color = '#F5210D')
ax.plot(hourly_mean.index, hourly_mean['CAPE_dilute'], label = 'CAPE_dilute',marker="x", color = '#8BC34A')
ax.plot(hourly_mean.index, hourly_mean['dCAPE'], label = 'dCAPE',marker="x", color = '#9B59B6')
ax.plot(hourly_mean.index, hourly_mean['dCAPE_dilute'], label = 'dCAPE_dilute',marker="x", color = '#00897B')
ax.set_ylim([0,1.0])
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Number of Convective Events / Total Length of Dataset", color = 'black')
#ax.legend(frameon=False)
ax2=ax.twinx()
ax2.plot(hourly_mean.index, hourly_mean['proba'], '--', label = 'Random forest probability, unadjusted',color='#1f77b4',marker="^")
ax2.plot(hourly_mean.index, hourly_mean['proba_Milo'], '--', label = 'Random forest probability, adjusted',color='orange',marker="^")
ax2.set_ylabel("Probability of Convection",color='black')
ax2.set_ylim([0,1.0])
fig.legend(frameon=False, loc='upper right', bbox_to_anchor=(0.85, 0.93))

fig.savefig(path+'diurnal_cycle_proba_Milo_CAPE_thresholds_tzhang.png',dpi=500, facecolor='white', transparent=False)

#same figure, but zoomed in on specific features
fig,ax = plt.subplots(figsize=(12,8))
ax.plot(hourly_mean.index, hourly_mean[y], label = 'Observed',marker=".", color = 'black')
# ax.plot(hourly_mean.index, hourly_mean['predict'], label = 'Random forest count, unadjusted',marker=".", color = '#1f77b4')
# ax.plot(hourly_mean.index, hourly_mean['predict_Milo'], label = 'Random forest count, adjusted',marker=".", color = 'orange')
# ax.plot(hourly_mean.index, hourly_mean['CAPE'], label = 'CAPE',marker="x", color = '#F5210D')
# ax.plot(hourly_mean.index, hourly_mean['CAPE_dilute'], label = 'CAPE_dilute',marker="x", color = '#8BC34A')
ax.plot(hourly_mean.index, hourly_mean['dCAPE'], label = 'dCAPE',marker="x", color = '#9B59B6')
ax.plot(hourly_mean.index, hourly_mean['dCAPE_dilute'], label = 'dCAPE_dilute',marker="x", color = '#00897B')
ax.set_ylim([0,0.3])
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Number of Convective Events / Total Length of Dataset", color = 'black')
#ax.legend(frameon=False)
ax2=ax.twinx()
#ax2.plot(hourly_mean.index, hourly_mean['proba'], '--', label = 'Random forest probability, unadjusted',color='#1f77b4',marker="^")
ax2.plot(hourly_mean.index, hourly_mean['proba_Milo'], '--', label = 'Random forest probability, adjusted',color='orange',marker="^")
ax2.set_ylabel("Probability of Convection",color='black')
ax2.set_ylim([0,0.3])
fig.legend(frameon=False, loc='upper right', bbox_to_anchor=(0.85, 0.93))
fig.tight_layout()
fig.savefig(path+'diurnal_cycle_proba_Milo_CAPE_thresholds_tzhang_zoomed.png',dpi=500, facecolor='white', transparent=False)

# %%
