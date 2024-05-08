"""

Train random forest to predict onset of convection

"""
#%%
from rf_functions import *
import seaborn as sns


#%%
path = 'output/rf/'
make_folder(path)
SEED = 1

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

# split data into train and test 
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


#%% Optimize hyperparameters
#bayes_hyperparameter(x, y, x_u_train[x], y_u_train, SEED) #creates a csv file
hp = pd.read_csv('output/rf/hyperparameters.csv')#reads in the csv file

#%% Create random forest
rf=RandomForestClassifier(n_estimators = hp['n_estimators'][0],
                         max_features = hp['max_features'][0],
                         min_samples_split = hp['min_samples_split'][0],
                         min_samples_leaf = hp['min_samples_leaf'][0],
                         random_state=SEED,
                         n_jobs=-1)
rf.fit(x_u_train[x],y_u_train)

# save
joblib.dump(rf, path+"rf.joblib")

# %% Model results
#training dataset
model_scores(rf, x, y, r, train, x_u_train[x], y_u_train, path, filename = '')

#%%
#test dataset
model_scores(rf, x, y, r, test, x_u_test[x], y_u_test, path, filename = '')

#%%
# Scaling for raw probablities from ML model to account for undersampling
proba_full = rf.predict_proba(test[x])[:, 1]
proba_updated = proba_full/(r + (1-r)*proba_full)
color1 = '#1f77b4'
color2 = '#8FBD70'

#%%
#Reliability curve (Miloshevich adjustment) with histogram in the background
pt = plt.figure(figsize=(6,5))
plt.plot([0, 1], [0, 1], "k--",label="Perfect reliability")
plt.ylabel("Observed frequency in bin")
plt.xlabel("Predicted probability in bin")
fraction_of_positives_rfc, mean_predicted_value_rfc = calibration_curve(test[y], proba_full, n_bins=20)
plt.plot(mean_predicted_value_rfc, fraction_of_positives_rfc, "s-", color = color1, label="%s" % ('Model, unscaled'))
fraction_of_positives_rfc, mean_predicted_value_rfc = calibration_curve(test[y], proba_updated, n_bins=20)
plt.plot(mean_predicted_value_rfc, fraction_of_positives_rfc,  "s-", color = color2, label="%s" % ('Model, scaled'))
plt.legend(loc="upper left", bbox_to_anchor=(0.2, 0.97), frameon=False)
plt.twinx()
#histogram
predictions = proba_updated
truth = test[y]
rounded = [round(x) for x in predictions]
incorrect = predictions[rounded != truth]
bins = np.linspace(0,1,21)
plt.hist(predictions, bins = bins, rwidth=0.8, alpha = 0.3, color = 'gray', label = 'correct')
plt.hist(incorrect, bins = bins, rwidth=0.8, alpha = 0.6, color = 'gray', hatch='//', label = 'incorrect')
plt.ylabel("Sample size per bin", color = 'gray')
plt.ylim([-20,600])
plt.yticks([0,100,200,300,400,500,600], labels = [0,100,200,300,400,500,5000])
plt.legend(loc="center right", frameon=False)
pt.tight_layout()
pt.savefig(path+'reliability_curve_balanced_Miloshevich_update.png',dpi=500, facecolor='white', transparent=False)

#%%
#Reliability curve (Miloshevich adjustment) for balanced (undersampled) predictions on balanced dataset
proba_undersampled = rf.predict_proba(x_u_test[x])[:, 1]

pt = plt.figure(figsize=(6,5))
plt.plot([0, 1], [0, 1], "k--",label="Perfect reliability")
plt.ylabel("Observed frequency in bin")
plt.xlabel("Predicted probability in bin")
fraction_of_positives_rfc, mean_predicted_value_rfc = calibration_curve(y_u_test, proba_undersampled, n_bins=20)
plt.plot(mean_predicted_value_rfc, fraction_of_positives_rfc, "s-", color = color1, label="%s" % ('Undersampled'))
plt.legend(loc="upper left", bbox_to_anchor=(0.2, 0.97), frameon=False)
plt.twinx()
#histogram
predictions = proba_undersampled
truth = y_u_test
rounded = [round(x) for x in predictions]
incorrect = predictions[rounded != truth]
bins = np.linspace(0,1,21)
plt.hist(predictions, bins = bins, rwidth=0.8, alpha = 0.3, color = 'gray', label = 'correct')
plt.hist(incorrect, bins = bins, rwidth=0.8, alpha = 0.6, color = 'gray', hatch='//', label = 'incorrect')
plt.ylabel("Sample size per bin", color = 'gray')
plt.ylim([-20,600])
plt.yticks([0,100,200,300,400,500,600], labels = [0,100,200,300,400,500,'X'])
plt.legend(loc="center right", frameon=False)
pt.tight_layout()
pt.savefig(path+'reliability_curve_balanced_Miloshevich_update_undersampled_dataset.png',dpi=500, facecolor='white', transparent=False)


#%%
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

#fix time to local time (SGP is 5 hours behind UTC)
lst_time = np.arange(24) - 5
lst_time[lst_time < 0] += 24
hourly_mean['hour_lst'] = lst_time
hourly_mean = hourly_mean.sort_values('hour_lst')

fig,ax = plt.subplots(figsize=(7,4.5))
ax.plot(hourly_mean.hour_lst, hourly_mean[y], label = 'Observed frequency',marker=".", color = 'black')
ax.plot(hourly_mean.hour_lst, hourly_mean['predict'], label = 'Model frequency, unadjusted',marker=".", color = color1)
ax.plot(hourly_mean.hour_lst, hourly_mean['predict_Milo'], label = 'Model frequency, adjusted',marker=".", color = color2)
ax.set_ylim([0,0.3])
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Probability or frequency of convection", color = 'black')
ax.plot(hourly_mean.hour_lst, hourly_mean['proba'], '--', label = 'Model probability, unadjusted',color=color1,marker="^")
ax.plot(hourly_mean.hour_lst, hourly_mean['proba_Milo'], '--', label = 'Model probability, adjusted',color=color2,marker="^")
fig.legend(frameon=False, loc='upper left', bbox_to_anchor=(0.11, 0.96), ncol = 2)
fig.tight_layout()
fig.savefig(path+'diurnal_cycle_proba_Milo.png',dpi=500, facecolor='white', transparent=False)

#%%Time Series
w = d[(d['timef'].dt.year ==2017)]
w = w.reset_index(drop=True)
pt = plt.figure(figsize=(9,5))#, gridspec_kw={'height_ratios': [1, 1, 3]}

plt.subplot(511)
plt.fill_between(w['timef'], w[y], step = 'post', label = 'Observed', color = 'black', edgecolor = 'none')
plt.ylabel("Observed\nEvents")
plt.tick_params(left = False, right = False , labelleft = False ,labelbottom = False, bottom = False)
plt.xlim([w['timef'][0], w['timef'][w.index[-1]]])

plt.subplot(512)
plt.fill_between(w['timef'], w['predict'],step = 'post', label = 'Prediction', color = color1,  edgecolor = 'none')
plt.ylabel("Unscaled\nPredicted\nEvents")
plt.tick_params(left = False, right = False , labelleft = False ,labelbottom = False, bottom = False)
plt.xlim([w['timef'][0], w['timef'][w.index[-1]]])

plt.subplot(513)
plt.fill_between(w['timef'], w['predict_Milo'],step = 'post', label = 'Prediction', color = color2,  edgecolor = 'none')
plt.ylabel("Scaled\nPredicted\nEvents")
plt.tick_params(left = False, right = False , labelleft = False ,labelbottom = False, bottom = False)
plt.xlim([w['timef'][0], w['timef'][w.index[-1]]])

plt.subplot(514)
plt.plot(w['timef'], w['proba'], color = color1, linewidth = 1, label = 'Probability')
plt.xlim([w['timef'][0], w['timef'][w.index[-1]]])
plt.tick_params(labelbottom = False, bottom = False)
plt.ylabel("Unscaled\nProbability")
plt.ylim([-0.1,1.1])

plt.subplot(515)
plt.plot(w['timef'], w['proba_Milo'], color = color2, linewidth = 1,label = 'Probability')
plt.xlim([w['timef'][0], w['timef'][w.index[-1]]])
plt.ylabel("Scaled\nProbability")
plt.ylim([-0.1,1.1])

plt.xlabel("Time")
pt.tight_layout()
pt.savefig(path+'timeseries_Milo.png',dpi=800, facecolor='white', transparent=False)

