"""

Functions used by rf.py to run random forest model

"""

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import os
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score
from sklearn.metrics import make_scorer
from sklearn.metrics import recall_score, precision_score, f1_score, brier_score_loss, PrecisionRecallDisplay
from sklearn.inspection import permutation_importance
from imblearn.under_sampling import RandomUnderSampler
from skopt import BayesSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn import tree
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from seaborn import kdeplot
import csv
from datetime import datetime, timedelta
import joblib

labels = {'LH': 'Latent heat',
        'SH': 'Sensible heat',
        'T_srf':'T surface', 
        'rh':'RH surface',
        'CIN_mu': 'CIN', 
        'LCL': 'LCL', 
        'PW': 'PW',
        'PW_q': 'PW',
        'undiluteCAPE_zm_mu':'CAPE',
        'CAPE_metpy_mu':'CAPE metpy mu',
        'diluteCAPE_zm_mu':'dilute CAPE',
        'dCAPE_zm_undilute_tzhang':'dCAPE',
        'dCAPE_zm_dilute_tzhang':'dilute dCAPE',
        'T_low': 'T low', 
        'T_mid': 'T mid',
        'T_high': 'T high',
        'q_low': 'q low', 
        'q_mid': 'q mid', 
        'q_high': 'q high', 
        'rh_low': 'RH low', 
        'rh_mid': 'RH mid', 
        'rh_high': 'RH high', 
        'q_adv_h_low': 'Adv. q low',
        'q_adv_h_mid': 'Adv. q mid',
        'q_adv_h_high':'Adv. q high',
        'q_adv_v_low': 'Adv.v q low',
        'q_adv_v_mid': 'Adv.v q mid',
        'q_adv_v_high': 'Adv.v q high',        
        's_adv_h_low':'Adv. s low',
        's_adv_h_mid':'Adv. s mid', 
        's_adv_h_high':'Adv. s high',
        's_adv_v_low':'Adv.v s low',
        's_adv_v_mid':'Adv.v s mid', 
        's_adv_v_high':'Adv.v s high',
        'T_adv_v_low':'Adv.v T low',
        'T_adv_v_mid':'Adv.v T mid', 
        'T_adv_v_high':'Adv.v T high',
        'adiab_adv_v_low':'Adv.v adiab low',
        'adiab_adv_v_mid':'Adv.v adiab mid', 
        'adiab_adv_v_high':'Adv.v adiab high',
        'omega_low':'ω low',
        'omega_mid':'ω mid', 
        'omega_high':'ω high',
        'omega_500_400':'omega_500_400',
        'mse_low': 'MSE low',
        'mse_mid': 'MSE mid',
        'mse_high': 'MSE high',
        'mse_srf': 'MSE srf',
        'shear_low':'Shear low', 
        'shear_mid':'Shear mid', 
        'shear_high':'Shear high',
        'random1': 'Random 1',
        'random2': 'Random 2',
        'random3': 'Random 3',
}



def make_folder(path):
    try:
        os.mkdir(path)
    except OSError:
        print ("Directory %s already exists" % path)
    else:
        print ("Successfully created the directory %s " % path)

'''Bayesian optimization of hyperparameters'''

def bayes_hyperparameter(x,y,x_u, y_u, SEED):
    #max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    #max_depth.append(None)

    # Create the hyperparameter grid
    params = {'n_estimators': (50,500),       # Number of trees in random forest
              'max_features': (8,len(x)),       # Number of features to consider at every split, default = 'sqrt'
              'min_samples_split': (2,4),      # Minimum number of samples required to split a node
              'min_samples_leaf': (1,4),# Minimum number of samples required at each leaf node
            }
    # first create the base model to tune
    rf = RandomForestClassifier(random_state=SEED, n_jobs=-1)
    # define evaluation
    cv = RepeatedStratifiedKFold(n_splits=4, n_repeats=4, random_state=SEED)
    # define the search
    search = BayesSearchCV(estimator=rf, search_spaces=params, scoring=make_scorer(f1_score), n_iter = 200, n_jobs=-1, cv=cv)
    # perform the search
    search.fit(x_u,y_u)
    # report the best result
    print(search.best_params_)
    hp = search.best_params_
    with open("output/rf/hyperparameters.csv", "w") as outfile:
        csvwriter = csv.writer(outfile)
        csvwriter.writerow(dict(hp))
        csvwriter.writerow(dict(hp).values())


'''Model Scores'''

def model_scores(rf, x, y, r, test, x_u_test, y_u_test, path, filename = ''):
        predict_y = rf.predict(test[x])
        proba = rf.predict_proba(test[x])[:, 1]
        proba_adjust = proba/(r + (1-r)*proba)
        predict_y_Milo = np.multiply([proba_adjust > 0.5],1)[0]
        predict_y_u = rf.predict(x_u_test)
        proba_u = rf.predict_proba(x_u_test)[:, 1]


        scores = {'Dataset': ['Unbalanced', 'Unbalanced Milo Adjustment','Balanced'],
        'Precision': [precision_score(test[y], predict_y), 
                        precision_score(test[y], predict_y_Milo), 
                        precision_score(y_u_test, predict_y_u)],
        'Recall': [recall_score(test[y], predict_y), 
                        recall_score(test[y], predict_y_Milo),
                        recall_score(y_u_test, predict_y_u)],
        'F1': [f1_score(test[y], predict_y), 
                        f1_score(test[y], predict_y_Milo), 
                        f1_score(y_u_test, predict_y_u)],
        'Brier': [brier_score_loss(test[y], proba), 
                        brier_score_loss(test[y], proba_adjust), 
                        brier_score_loss(y_u_test, proba_u)]
        }

        df = pd.DataFrame(data=scores)
        df.to_csv(path+'scores'+filename+'.csv') 


        return df

'''Feature Importance'''

def feature_importance(rf, x, y, x_set, y_set, path, filename= ''):
    pt = plt.figure(figsize=(7,6))
    ######### IMPURITY IMPORTANCE ########
    importances = rf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rf.estimators_],axis=0)
    indices = np.argsort(importances)[::-1]

    #ordered and named importances
    var = list(x)
    var_ordered = [var[i] for i in indices]
    var_labels = [labels[i] for i in var_ordered]

    # Plot the impurity-based feature importances of the forest
    plt.subplot(211)
    plt.ylabel("Impurity \n Feature Importance")
    plt.bar(range(x.shape[0]), importances[indices],
            color="lightseagreen", yerr=std[indices], align="center")
    plt.xticks(range(x.shape[0]), var_labels,rotation=45, horizontalalignment='right',rotation_mode='anchor')
    plt.xlim([-1, x.shape[0]])
    plt.ylim([-0.04, 0.5])
    ######### PERMUTATION IMPORTANCE ############
    perm = permutation_importance(rf, x_set, y_set, n_repeats=10, n_jobs = -1, random_state=0)                         
    importances = perm.importances_mean
    std = perm.importances_std
    indices = np.argsort(importances)[::-1]

    #ordered and named importances
    var = list(x)
    var_ordered = [var[i] for i in indices]
    var_labels = [labels[i] for i in var_ordered]

    # Plot the permutation-based feature importances of the forest
    plt.subplot(212)
    plt.ylabel("Permutation \n Feature Importance")
    plt.bar(range(x.shape[0]), importances[indices],
            color="lightseagreen", yerr=std[indices], align="center")
    plt.xticks(range(x.shape[0]), var_labels,rotation=45, horizontalalignment='right',rotation_mode='anchor')
    plt.xlim([-1, x.shape[0]])
    pt.tight_layout()
    pt.savefig(path+'feature_importance'+filename+'.png',dpi=500, facecolor='white', transparent=False)

'''Reliability Curve'''

def reliability_curve(rf,x,y,test,path,filename = ''):
        pt = plt.figure(figsize=(5,4))
        plt.plot([0, 1], [0, 1], "k--",label="Perfectly calibrated")
        plt.ylabel("Observed frequency in bin")
        plt.xlabel("Predicted probability in bin")
        #plt.title('Calibration plot (reliability curve)')

        prob_pos_rfc = rf.predict_proba(test[x])[:, 1]
        fraction_of_positives_rfc, mean_predicted_value_rfc = calibration_curve(test[y], prob_pos_rfc, n_bins=10)
        plt.plot(mean_predicted_value_rfc, fraction_of_positives_rfc, "s-", label="%s" % ('Random forest'))

        plt.legend(loc="lower right", frameon=False)
        pt.tight_layout()
        pt.savefig(path+'reliability_curve'+filename+'.png',dpi=500, facecolor='white', transparent=False)

'''Histogram Plot'''

def histogram_plot(rf,x,y,test,path,filename = ''):
        # make probability predictions with the model
        predictions = rf.predict_proba(test[x])[:,1]
        truth = test[y]
        rounded = [round(x) for x in predictions]
        incorrect = predictions[rounded != truth]

        pt = plt.figure(figsize=(8,6))
        bins = np.linspace(0,1,11)
        fig, ax = plt.subplots(nrows=1, ncols=1)
        pred_hist = ax.hist(predictions, bins = bins, rwidth=0.8, label = 'correct')
        incorrect_hist = ax.hist(incorrect, bins = bins, rwidth=0.8, label = 'incorrect')
        ax.set_ylim([0,2000])
        ax.set_xlabel('Predicted probability')
        ax.set_ylabel('Count')
        ax.legend(loc = 'upper right', frameon = False)
        fig.tight_layout()
        fig.savefig(path+'histogram'+filename+'.png',dpi=500, facecolor='white', transparent=False)

        # prob_pred = np.empty(10)
        # prob_true = np.empty(10)
        # for i in range(10):
        # subset_pred = predictions[((predictions>bins[i]) & (predictions<bins[i+1]))]
        # subset_true = truth.iloc[((predictions>bins[i]) & (predictions<bins[i+1]))]
        # prob_pred[i] = np.mean(subset_pred)
        # prob_true[i] = np.mean(subset_true)
        
        # fig, ax = plt.subplots(nrows=1, ncols=1)
        # ax.plot([0,1], [0,1],'k--')
        # ax.plot(prob_pred, prob_true, marker = 'o')
        # ax.set_xlabel('Predicted probability in bin')
        # ax.set_ylabel('Number of true events/Total number of events in bin')

'''Reliability Curve with Histogram'''

def reliability_curve_hist(rf,x,y,test,path,filename = ''):
        pt = plt.figure(figsize=(10,7))

        plt.subplot(211)
        plt.plot([0, 1], [0, 1], "k--",label="Perfectly calibrated")
        plt.ylabel("Observed frequency")
        prob_pos_rfc = rf.predict_proba(test[x])[:, 1]
        fraction_of_positives_rfc, mean_predicted_value_rfc = calibration_curve(test[y], prob_pos_rfc, n_bins=10)
        plt.plot(mean_predicted_value_rfc, fraction_of_positives_rfc, "s-", label="%s" % ('Random forest'))
        plt.legend(loc="lower right", frameon=False)

        plt.subplot(212)
        # make probability predictions with the model
        predictions = rf.predict_proba(test[x])[:,1]
        truth = test[y]
        rounded = [round(x) for x in predictions]
        incorrect = predictions[rounded != truth]
        bins = np.linspace(0,1,11)
        plt.hist(predictions, bins = bins, rwidth=0.8, label = 'correct')
        plt.hist(incorrect, bins = bins, rwidth=0.8, label = 'incorrect')
        plt.xlabel("Predicted probability")
        plt.ylabel("Count")

        pt.tight_layout()
        pt.savefig(path+'reliability_curve_hist'+filename+'.png',dpi=500, facecolor='white', transparent=False)


'''Kernel Density Plot'''

def kde_plot(rf,x,test,path,filename = ''):
        pt = plt.figure(figsize=(8,6))
        prob_pos_rfc = rf.predict_proba(test[x])[:, 1]
        kdeplot(data=prob_pos_rfc, bw_adjust = 0.4)
        plt.xlabel("Predicted probability of convection")
        plt.ylim([0,2])
        pt.tight_layout()
        pt.savefig(path+'kde_plot'+filename+'.png',dpi=500, facecolor='white', transparent=False)

'''Precision Recall Curve'''

def PR_curve(rf, x, y, test, path, filename= ''):
        pred_y = rf.predict_proba(test[x])[:,1]
        pt = plt.figure(figsize=(8,6))
        PrecisionRecallDisplay.from_predictions(test[y], pred_y, name = 'Random Forest')
        pt.tight_layout()
        pt.savefig(path+'PR_curve'+filename+'.png',dpi=500, facecolor='white', transparent=False)

'''Round time to the nearest hour'''

def hour_rounder(t):
    # Rounds to nearest hour by adding a timedelta hour if minute >= 30
    return (t.replace(second=0, microsecond=0, minute=0, hour=t.hour)+timedelta(hours=t.minute//30))


'''Diurnal Plot'''

def diurnal_plot_freq(rf, x, y, test, path, filename= ''):
        d = test.copy()
        d['predict'] = rf.predict(test[x])
        d['proba'] = rf.predict_proba(test[x])[:,1]
        d['timef'] = pd.to_datetime(d['time'])
        d['hour'] = [hour_rounder(t) for t in d['timef']]
        times = d['hour'].dt.hour
        hourly_sum = d.groupby(times).sum()

        pt = plt.figure(figsize=(8,6))
        plt.plot(hourly_sum.index, hourly_sum[y], label = 'Observed')
        plt.plot(hourly_sum.index, hourly_sum['predict'], label = 'Random forest')
        plt.xlabel("Hour of Day")
        plt.ylabel("Number of Convective Events")
        plt.legend(frameon=False)
        pt.tight_layout()
        pt.savefig(path+'diurnal_cycle_freq'+filename+'.png',dpi=500, facecolor='white', transparent=False)

'''Diurnal Plot with Probability Axis'''

def diurnal_plot_proba_freq(rf, x, y, test, path, filename= ''):
        d = test.copy()
        d['predict'] = rf.predict(test[x])
        d['proba'] = rf.predict_proba(test[x])[:,1]
        d['timef'] = pd.to_datetime(d['time'])
        d['hour'] = [hour_rounder(t) for t in d['timef']]
        times = d['hour'].dt.hour
        hourly_sum = d.groupby(times).sum()
        hourly_mean = d.groupby(times).mean()

        fig,ax = plt.subplots(figsize=(8,6))
        ax.plot(hourly_sum.index, hourly_sum[y], label = 'Observed')
        ax.plot(hourly_sum.index, hourly_sum['predict'], label = 'Random forest prediction (0 or 1)')
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Number of Convective Events")
        #ax.legend(frameon=False)
        ax2=ax.twinx()
        ax2.plot(hourly_sum.index, hourly_mean['proba'], label = 'Random forest probability',color="red",marker="o")
        ax2.set_ylabel("Probability of Convection",color="red")
        fig.legend(frameon=False, loc='upper right', bbox_to_anchor=(0.9, 0.93))
        fig.tight_layout()
        fig.savefig(path+'diurnal_cycle_proba_freq'+filename+'.png',dpi=500, facecolor='white', transparent=False)

'''Diurnal Plot with Probability Axis'''

def diurnal_plot_proba(rf, x, y, test, path, filename= ''):
        d = test.copy()
        d['predict'] = rf.predict(test[x])
        d['proba'] = rf.predict_proba(test[x])[:,1]
        d['timef'] = pd.to_datetime(d['time'])
        d['hour'] = [hour_rounder(t) for t in d['timef']]
        times = d['hour'].dt.hour
        hourly_sum = d.groupby(times).sum()
        hourly_mean = d.groupby(times).mean()

        fig,ax = plt.subplots(figsize=(6,4))
        ax.plot(hourly_sum.index, hourly_sum[y], label = 'Observed',marker=".", color = '#2ca02c')
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Number of Convective Events", color = '#2ca02c')
        #ax.legend(frameon=False)
        ax2=ax.twinx()
        ax2.plot(hourly_sum.index, hourly_mean['proba'], label = 'Random forest',color='#1f77b4',marker="o")
        ax2.set_ylabel("Probability of Convection",color='#1f77b4')
        fig.legend(frameon=False, loc='upper right', bbox_to_anchor=(0.85, 0.93))
        fig.tight_layout()
        fig.savefig(path+'diurnal_cycle_proba'+filename+'.png',dpi=500, facecolor='white', transparent=False)

'''Timeseries'''

def timeseries(rf, x, y, test, path, filename= ''):
        
        d = test.copy()
        d['predict'] = rf.predict(test[x])
        d['proba'] = rf.predict_proba(test[x])[:,1]
        d['timef'] = pd.to_datetime(d['time'])
        w = d[(d['timef'].dt.year ==2015)]
        w = w.reset_index(drop=True)

        pt = plt.figure(figsize=(12,6))#, gridspec_kw={'height_ratios': [1, 1, 3]}
        plt.subplot(311)
        plt.bar(w['timef'], w[y], label = 'Observed', width = 0.06)
        #plt.plot(w['timef'], w[y], color = 'red', linewidth=0.01)
        plt.ylabel("Observed\nEvents")
        plt.tick_params(left = False, right = False , labelleft = False ,labelbottom = False, bottom = False)
        plt.xlim([w['timef'][0], w['timef'][w.index[-1]]])
        plt.subplot(312)
        plt.bar(w['timef'], w['predict'],label = 'Prediction', width = 0.06)
        #plt.plot(w['timef'], w['predict'], linewidth=0.05, color = 'red')
        plt.ylabel("Predicted\nEvents")
        plt.tick_params(left = False, right = False , labelleft = False ,labelbottom = False, bottom = False)
        plt.xlim([w['timef'][0], w['timef'][w.index[-1]]])
        plt.subplot(313)
        plt.plot(w['timef'], w['proba'], color = 'black', label = 'Probability')
        plt.xlabel("Time")
        plt.xlim([w['timef'][0], w['timef'][w.index[-1]]])
        plt.ylabel("Probability")
        plt.ylim([0,1])
        pt.tight_layout()
        pt.savefig(path+'timeseries'+filename+'.png',dpi=800, facecolor='white', transparent=False)
