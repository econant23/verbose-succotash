close all

%% User Block: Select predictor and target for 1-D analyses
pred_names = {'Avg_Atom_Mass', 'Mean_Dopant_Valence', 'Mean_Other_Valence',...
    'Char_Size','Char_Aspect','Mean_Separation'};
targ_names = {'max_T','max_zT','max_sigma','max_S','max_kappa','corr_1',...
    'corr_2','slope_1','slope_2'};

scaleme = [0 0 0 1 1 1];

mypred = 'Avg_Atom_Mass';
mytarg = 'max_zT';

%% Read In Data File
filename = 'TEDATA.csv';
tedata = importdata(filename);
data_names = tedata.colheaders;
data_numbs = tedata.data;
numdata = size(data_numbs,1);

pred_arr = data_numbs(:,ismember(data_names,pred_names));
targ_arr = data_numbs(:,ismember(data_names,targ_names));
numpred = size(pred_arr,2);
numtarg = size(targ_arr,2);

%% Split data into classes
medians = median(targ_arr,1);
quart25 = prctile(targ_arr,25,1);
quart75 = prctile(targ_arr,75,1);

class2_arr = zeros(size(targ_arr));
class3_arr = zeros(size(targ_arr));

class2_arr(targ_arr > repmat(medians,numdata,1)) = 1;
class3_arr(targ_arr > repmat(quart25,numdata,1)) = 1;
class3_arr(targ_arr > repmat(quart75,numdata,1)) = 2;

%% Renormalize Predictors as Logarithms
norm_arr = pred_arr;
for i = 1:numpred
    if scaleme(i) == 1
        norm_arr(:,i) = log10(norm_arr(:,i));
    end
end
pred_arr = norm_arr;

%% Make Linear Regression Between User Predictor and Target
figure(1)
plot(pred_arr(:,ismember(pred_names,mypred)), ...
     targ_arr(:,ismember(targ_names,mytarg)),'.k')
set(gcf,'color','w') 
hx=xlabel(mypred);
hy=ylabel(mytarg);
set(hx,'interpreter','none') 
set(hy,'interpreter','none')

rnge1 = 0.9*min(pred_arr(:,ismember(pred_names,mypred)));
rnge2 = 1.1*max(pred_arr(:,ismember(pred_names,mypred)));

linregmat = [pred_arr(:,ismember(pred_names,mypred)), ones(numdata,1)];
linregpar = (linregmat'*linregmat)\eye(2)*linregmat'*...
    targ_arr(:,ismember(targ_names,mytarg));

data_act = targ_arr(:,ismember(targ_names,mytarg));
data_mod = linregpar(1).*pred_arr(:,ismember(pred_names,mypred))+linregpar(2);
coefdet = 1 - sum((data_act-data_mod).^2) / sum((data_act-mean(data_act)).^2);

hold on
plot(linspace(rnge1,rnge2),linregpar(1)*linspace(rnge1,rnge2)+linregpar(2),'--k')
eqstr = ['y=(', num2str(linregpar(1)),')*x + (', num2str(linregpar(2)),')'];
r2str = ['R^2=(',num2str(coefdet),')'];
t=text(0.01,0.97,eqstr,'Units','normalized');
t.FontWeight = 'bold';
t=text(0.01,0.92,r2str,'Units','normalized');
t.FontWeight = 'bold';

%% Make 2-class Boxplots for User Predictor Sorted by User Target Classes
figure(2)
boxplot([norm_arr(class2_arr(:,ismember(targ_names,mytarg))==0, ismember(pred_names,mypred));...
    norm_arr(class2_arr(:,ismember(targ_names,mytarg))==1, ismember(pred_names,mypred))], ...
    [zeros(sum(class2_arr(:,ismember(targ_names,mytarg))==0),1);...
        ones(sum(class2_arr(:,ismember(targ_names,mytarg))==1),1)])
set(gcf,'color','w')
hx=xlabel([mytarg,' by class']);
hy=ylabel(mypred);
set(hx,'interpreter','none')
set(hy,'interpreter','none')

[~,p_val] = ttest2( norm_arr(class2_arr(:,ismember(targ_names,mytarg))==0), ...
    norm_arr(class2_arr(:,ismember(targ_names,mytarg))==1) );
pvstr = ['p-val=(',num2str(p_val),')'];
t=text(0.01,0.97,pvstr,'Units','normalized');
t.FontWeight = 'bold';

%% Make 3-class Boxplots for User Predictor Sorted by User Target Classes
figure(3)

boxplot([norm_arr(class3_arr(:,ismember(targ_names,mytarg))==0, ismember(pred_names,mypred));...
    norm_arr(class3_arr(:,ismember(targ_names,mytarg))==1, ismember(pred_names,mypred)); ...
    norm_arr(class3_arr(:,ismember(targ_names,mytarg))==2, ismember(pred_names,mypred))], ...
    [zeros(sum(class3_arr(:,ismember(targ_names,mytarg))==0),1);...
        ones(sum(class3_arr(:,ismember(targ_names,mytarg))==1),1);...
        2*ones(sum(class3_arr(:,ismember(targ_names,mytarg))==2),1)])
%}
p_val = anova1([norm_arr(class3_arr(:,ismember(targ_names,mytarg))==0, ismember(pred_names,mypred));...
    norm_arr(class3_arr(:,ismember(targ_names,mytarg))==1, ismember(pred_names,mypred)); ...
    norm_arr(class3_arr(:,ismember(targ_names,mytarg))==2, ismember(pred_names,mypred))], ...
    [zeros(sum(class3_arr(:,ismember(targ_names,mytarg))==0),1);...
        ones(sum(class3_arr(:,ismember(targ_names,mytarg))==1),1);...
        2*ones(sum(class3_arr(:,ismember(targ_names,mytarg))==2),1)],'off');
pvstr = ['p-val=(',num2str(p_val),')'];
set(gcf,'color','w')
hx=xlabel([mytarg,' by class']);
hy=ylabel(mypred);
set(hx,'interpreter','none')
set(hy,'interpreter','none')
t=text(0.01,0.97,pvstr,'Units','normalized');
t.FontWeight = 'bold';

%% Make Hyperlinear Regressor for all Predictors (with intercept)
figure(4)

linregmat = [pred_arr, ones(numdata,1)];
linregpar = (linregmat'*linregmat)\eye(size(pred_arr,2)+1)*linregmat'*...
    targ_arr(:,ismember(targ_names,mytarg));

data_act = targ_arr(:,ismember(targ_names,mytarg));
data_mod = zeros(size(data_act));

for k = 1:numdata
    data_cur = pred_arr(k,:);
    data_mod(k) = data_cur*linregpar(1:end-1)+linregpar(end);
end
coefdet = 1 - sum((data_act-data_mod).^2) / sum((data_act-mean(data_act)).^2);
r2str = ['R^2 = (',num2str(coefdet),')'];

rnge1 = 0.9*min(targ_arr(:,ismember(targ_names,mytarg)));
rnge2 = 1.1*max(targ_arr(:,ismember(targ_names,mytarg)));

plot(data_act,data_mod,'.k')
set(gcf,'color','white')
hold on
plot(linspace(rnge1,rnge2),linspace(rnge1,rnge2),'--k')
hx=xlabel(['actual ',mytarg]);
hy=ylabel(['predicted ',mytarg]);
set(hx,'interpreter','none')
set(hy,'interpreter','none')

t=text(0.01,0.97,r2str,'Units','normalized');
t.FontWeight = 'bold';

%% Make Hyperlinear Regressor for all Predictors (without intercept)
figure(5)

linregmat = pred_arr;
linregpar = (linregmat'*linregmat)\eye(size(pred_arr,2))*linregmat'*...
    targ_arr(:,ismember(targ_names,mytarg));

data_act = targ_arr(:,ismember(targ_names,mytarg));
data_mod = zeros(size(data_act));

for k = 1:numdata
    data_cur = pred_arr(k,:);
    data_mod(k) = data_cur*linregpar;
end
coefdet = 1 - sum((data_act-data_mod).^2) / sum((data_act-mean(data_act)).^2);
r2str = ['R^2 = (',num2str(coefdet),')'];

rnge1 = 0.9*min(targ_arr(:,ismember(targ_names,mytarg)));
rnge2 = 1.1*max(targ_arr(:,ismember(targ_names,mytarg)));

plot(data_act,data_mod,'.k')
set(gcf,'color','white')
hold on
plot(linspace(rnge1,rnge2),linspace(rnge1,rnge2),'--k')
hx=xlabel(['actual ',mytarg]);
hy=ylabel(['predicted ',mytarg]);
set(hx,'interpreter','none')
set(hy,'interpreter','none')

t=text(0.01,0.97,r2str,'Units','normalized');
t.FontWeight = 'bold';

%% Make Multiquadratic Model with Interactions for all Predictors
figure(6)

linregmat1 = [pred_arr, ones(numdata,1)];
linregmat2 = zeros(numdata, numpred + nchoosek(numpred,2));
nanmatrix2 = triu(ones(numpred));
nanmatrix2(nanmatrix2==0) = nan;

for m = 1:numdata
    newmatrix = pred_arr(m,:)'*pred_arr(m,:).*nanmatrix2;
    newmatrix = newmatrix(:)';
    newmatrix(isnan(newmatrix)) = [];
    linregmat2(m,:) = newmatrix;
end

linregmat = [linregmat2,linregmat1];

linregpar = (linregmat'*linregmat)\eye(size(linregmat,2))*linregmat'*...
    targ_arr(:,ismember(targ_names,mytarg));

data_act = targ_arr(:,ismember(targ_names,mytarg));
data_mod = linregmat*linregpar;

coefdet = 1 - sum((data_act-data_mod).^2) / sum((data_act-mean(data_act)).^2);
r2str = ['R^2 = (',num2str(coefdet),')'];

rnge1 = 0.9*min(targ_arr(:,ismember(targ_names,mytarg)));
rnge2 = 1.1*max(targ_arr(:,ismember(targ_names,mytarg)));

plot(data_act,data_mod,'.k')
set(gcf,'color','white')
hold on
plot(linspace(rnge1,rnge2),linspace(rnge1,rnge2),'--k')
hx=xlabel(['actual ',mytarg]);
hy=ylabel(['predicted ',mytarg]);
set(hx,'interpreter','none')
set(hy,'interpreter','none')

t=text(0.01,0.97,r2str,'Units','normalized');
t.FontWeight = 'bold';

%% Perform Lasso Regression and Observe Convergence
figure(7)
[lassomat, fit] = lasso(pred_arr,targ_arr(:,ismember(targ_names,mytarg)));
lassomat = lassomat';
lambda = fit.Lambda;

semilogx(lambda, lassomat, '-')
set(gcf,'color','w')
xlabel('\lambda')
ylabel('\phi_i')
l=legend(pred_names,'location','southwest');
set(l,'interpreter','none')

%% Perform PCA Analysis and Visualize Clustering Behavior
figure(7)

[~, scr, var] = pca(pred_arr);
subplot(2,2,1)
semilogy([1:numpred],var, '-ok')
set(gcf,'color','white')
xlim([1,numpred])
ylim([0.8*min(var),1.2*max(var)])
xlabel('PCA No.')
ylabel('Explained Var.')

subplot(2,2,2)
plot(scr(class2_arr(:,ismember(targ_names,mytarg))==0,1), ...
    scr(class2_arr(:,ismember(targ_names,mytarg))==0,2), 'ob')
hold on
plot(scr(class2_arr(:,ismember(targ_names,mytarg))==1,1), ...
    scr(class2_arr(:,ismember(targ_names,mytarg))==1,2), 'or')
set(gcf,'color','w')
xlabel('PCA 1')
ylabel('PCA 2')

subplot(2,2,3)
plot(scr(class2_arr(:,ismember(targ_names,mytarg))==0,1), ...
    scr(class2_arr(:,ismember(targ_names,mytarg))==0,3), 'ob')
hold on
plot(scr(class2_arr(:,ismember(targ_names,mytarg))==1,1), ...
    scr(class2_arr(:,ismember(targ_names,mytarg))==1,3), 'or')
set(gcf,'color','w')
xlabel('PCA 1')
ylabel('PCA 3')

subplot(2,2,4)
plot(scr(class2_arr(:,ismember(targ_names,mytarg))==0,2), ...
    scr(class2_arr(:,ismember(targ_names,mytarg))==0,3), 'ob')
hold on
plot(scr(class2_arr(:,ismember(targ_names,mytarg))==1,2), ...
    scr(class2_arr(:,ismember(targ_names,mytarg))==1,3), 'or')
set(gcf,'color','w')
xlabel('PCA 2')
ylabel('PCA 3')

%% Perform Sequential Search Feature Selection
%{
figure(8)
flag = 1;
if flag == 1

repeats = 4;
numsim = 50;
fraction = 0.95;
cur_class = class2_arr(:,ismember(targ_names,mytarg));
        
% Run $repeats of simulations of size $numsims
summary = zeros(repeats,numpred);
for j = 1:repeats
    resmat = zeros(numsim,numpred);
    h = waitbar(0,'Running Sequential Selection');
    for k = 1:numsim
        waitbar(k/numsim,h)
        randind = randperm(numdata,round(fraction*numdata));
        newpred = pred_arr(randind,:);
        newclass = cur_class(randind,:);
        bestset = plusKminusRc(newpred,newclass);
        resmat(k,bestset) = 1;
    end
    summary(j,:) = sum(resmat,1)./numsim;
    close(h)
end

boxplot(summary)
set(gcf,'color','w')
xlabel('Predictors');
ylabel('Seq. Search Selection %');
t=text(0.01,0.97,mytarg,'Units','normalized');
t.FontWeight = 'bold';
set(t,'interpreter','none')
end
%}