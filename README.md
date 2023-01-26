# NNARX
The project uses a nonlinear autoregressive exogenous (NARX), model to make time-series prediction on data obtained from riverbank erosion combining reach-scale hydraulic and bank-scale erosion Nonlinear Multi Independent variables.

WHAT IS THIS REPOSITORY FOR?

The project uses a nonlinear autoregressive exogenous (NARX) model to make time-series prediction on data obtained from riverbank erosion combining reach-scale hydraulic and bank-scale erosion Nonlinear Multi Independent variables. The project consists of 1 dependent variable (erosion rate to the near-bank velocity) and 11 independent variables (multi factors to the erosion rate). Study area took place at Bernam River, Selangor, Malaysia where severe riverbank erosion occurred due to multiple reach-scale hydraulic and bank-scale factors.

![Picture1](https://user-images.githubusercontent.com/85818234/214748373-fda3a3d2-e05e-4980-bc25-89014e772199.jpg)

Figure 1: Eroded riverbank at Bernam River.


CODE DESCRIPTION:

The algorithm for NNARX has been developed in MATLAB software. The algorithm developed following the spatial and temporal limit of the present data for the model development and model validation. This algorithm allows any user to modify the algorithm based on the temporal and spatial limits of the sample size. 

*First, load the training data in a table form. Data for this project consist of 1 dependent variable (y) and 11 independent variables (x1 to x11). Use 10 neurons in the hidden layer:*

clear, close all
clc
 
load('C:\Users\Azlinda\Desktop\model1.mat');
rand('twister',1);
noHidden = 10;

a = allData';
a = a';
 
y = a(:,1);
x1 = a(:,2);
x2 = a(:,3);
x3 = a(:,4);
x4 = a(:,5);
x5 = a(:,6);
x6 = a(:,7);
x7 = a(:,8);
x8 = a(:,9);
x9 = a(:,10);
x10 = a(:,11);
x11 = a(:,12);
 
yt = y(4:398);
yt1 = y(3:397);
yt2 = y(2:396);
yt3 = y(1:395);
 
x1t1 = x1(3:397);
x1t2 = x1(2:396);
x1t3 = x1(1:395);
 
x2t1 = x2(3:397);
x2t2 = x2(2:396);
x2t3 = x2(1:395);
 
x3t1 = x3(3:397);
x3t2 = x3(2:396);
x3t3 = x3(1:395);
 
x4t1 = x4(3:397);
x4t2 = x4(2:396);
x4t3 = x4(1:395);
 
x5t1 = x5(3:397);
x5t2 = x5(2:396);
x5t3 = x5(1:395);
 
x6t1 = x6(3:397);
x6t2 = x6(2:396);
x6t3 = x6(1:395);
 
x7t1 = x7(3:397);
x7t2 = x7(2:396);
x7t3 = x7(1:395);
 
x8t1 = x8(3:397);
x8t2 = x8(2:396);
x8t3 = x8(1:395);
 
x9t1 = x9(3:397);
x9t2 = x9(2:396);
x9t3 = x9(1:395);
 
x10t1 = x10(3:397);
x10t2 = x10(2:396);
x10t3 = x10(1:395);
 
x11t1 = x11(3:397);
x11t2 = x11(2:396);
x11t3 = x11(1:395);
 
psi = [yt1, yt2, yt3, x1t1, x1t2, x1t3, x2t1, x2t2, x2t3, x3t1, x3t2, x3t3, x4t1, ....
    x4t2, x4t3, x5t1, x5t2, x5t3, x6t1, x6t2, x6t3, x7t1, x7t2, x7t3, ...
    x8t1, x8t2, x8t3, x9t1, x9t2, x9t3, x10t1, x10t2, x10t3, x11t1, ...
    x11t2, x11t3];
 
psi = psi';
yt = yt';


*Create the series-parallel NARX network using the function mapminmax:*

[psi, ps] = mapminmax(psi);
[yt, ys] = mapminmax(yt);
 
net = feedforwardnet(noHidden);
[net, tr] = train(net,psi,yt);
yhat = net(psi);
resid = yt - yhat;
 
yt_trn = yt(:,tr.trainInd);
yt_val = yt(:,tr.valInd);
yt_tst = yt(:,tr.testInd);
 
yhat_trn = yhat(:,tr.trainInd);
yhat_val = yhat(:,tr.valInd);
yhat_tst = yhat(:,tr.testInd);
 
resid_trn = resid(:,tr.trainInd);
resid_val = resid(:,tr.valInd);
resid_tst = resid(:,tr.testInd);


*For fitting test:*

%fitting test

figure, plot(yt_trn);    %real answer

hold on;

plot(yhat_trn,'r--');    %prediction (red)

hold off;
 
%fitting test
figure, plot(yt_tst);    %real answer
hold on;
plot(yhat_tst,'r--');    %prediction (red)
hold off;
 

*For model accuracy (r-squared) and MSE:*

rsq_trn = 100 * (1 - (sum(resid_trn .^2)/sum((yt_trn - mean(yt_trn)) .^2)))
rsq_tst = 100 * (1 - (sum(resid_tst .^2)/sum((yt_tst - mean(yt_tst)) .^2)))
mse_trn = mse(resid_trn)
mse_tst = mse(resid_tst)
 

psi_trn = psi(:,tr.trainInd)';
psi_tst = psi(:,tr.testInd)';
 
yt_trn = yt(:,tr.trainInd)';

rsq_trnANN = 100 * (1 - (sum(reverseResidTrnANN .^2)/sum((reverseOri_trn - mean(reverseOri_trn)) .^2)))


rsq_tstANN = 100 * (1 - (sum(reverseResidTstANN .^2)/sum((reverseOri_tst - mean(reverseOri_tst)) .^2)))


mse_reverseANN_trn = mse(reverseResidTrnANN)
mse_reverseANN_tst = mse(reverseResidTstANN)

 
figure, plot(reverseOri_tst);
hold on;
plot(reverseANN_tst,'r--');
hold off;


END



RESULTS (OSA PREDICTION PLOT):

*Model configurations tested and OSA prediction plots for both training and testing dataset for all models were generated based on time series erosion rate. OSA prediction plots were examined, and predicted dataset resembled closely to the actual dataset are deemed to be accurate. Both training and testing OSA prediction plots for all six model configurations show good agreement between the model prediction (blue line) and the actual data (red line), indicating the good predictive performance.*

![2a](https://user-images.githubusercontent.com/85818234/214755178-2967a2c4-df97-4399-bee2-f86c9fa3d28d.jpg)

Figure 2 (a): OSA prediction plot between predicted erosion rates with measured erosion rates for testing dataset.

![2b](https://user-images.githubusercontent.com/85818234/214755209-72c959c7-7ad4-4bab-884e-10a1f47cdd8b.jpg)

Figure 2(b): OSA prediction plot between predicted erosion rates with measured erosion rates for testing dataset.





