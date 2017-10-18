
flag1 = 1;
flag2 = 1;
flag34 = 1;
flag5 = 1;
flag6 = 1;
flag7 = 1;
flag8 = 1;
flag9 = 1;
flag10 = 1;

%% Read In Data File
filename = 'TEDATA.csv';
tedata = importdata(filename);
data_names = tedata.colheaders;
data_numbs = tedata.data;
numdata = size(data_numbs,1);

pred_names = {'Avg_Atom_Mass', 'Mean_Dopant_Valence', 'Mean_Other_Valence',...
    'Char_Size','Char_Aspect','Mean_Separation'};
targ_names = {'max_T','max_zT','max_sigma','max_S','max_kappa','corr_1',...
    'corr_2','slope_1','slope_2'};
scaleme = [0 0 0 1 1 1];

pred_arr = data_numbs(:,ismember(data_names,pred_names));
targ_arr = data_numbs(:,ismember(data_names,targ_names));

% Split Data into Classes
medians = median(targ_arr,1);
quart25 = prctile(targ_arr,25,1);
quart75 = prctile(targ_arr,75,1);

class2_arr = zeros(size(targ_arr));
class3_arr = zeros(size(targ_arr));

class2_arr(targ_arr > repmat(medians,numdata,1)) = 1;
class3_arr(targ_arr > repmat(quart25,numdata,1)) = 1;
class3_arr(targ_arr > repmat(quart75,numdata,1)) = 2;

% Renormalize some predictors with logarithm [Some predictors vary over 1000x]
norm_arr = pred_arr;
for i = 1:numpred
    if scaleme(i) == 1
        norm_arr(:,i) = log10(norm_arr(:,i));
    end
end
pred_arr = norm_arr;

numdata = size(pred_arr,1);
numpred = size(pred_arr,2);
numtarg = size(targ_arr,2);

%% FS Test 1: Two-Class T-Test Comparison for Each Predictor
if flag1 == 1
    test1 = zeros(numtarg,numpred);    
    for i = 1:numpred
        pred_cur = norm_arr(:,i);        
        for j = 1:numtarg
            class_cur = class2_arr(:,j);            
            [~,p_val] = ttest2( pred_cur(class_cur==0),pred_cur(class_cur==1));
            test1(j,i) = 1-p_val;
        end
    end    
end

test1

%% FS Test 2: Three-Class ANOVA Comparison for each Predictor
if flag2 == 1
    test2 = zeros(numtarg,numpred);    
    for i = 1:numpred
        pred_cur = norm_arr(:,i);        
        for j = 1:numtarg
            class_cur = class3_arr(:,j);            
            p_val = anova1( [pred_cur(class_cur==0);pred_cur(class_cur==1);pred_cur(class_cur==2)],...
                [zeros(sum(class_cur==0),1);ones(sum(class_cur==1),1);2.*ones(sum(class_cur==2),1)],'off');
            test2(j,i) = 1-p_val;
        end
    end    
end

test2

%% FS Tests 3 & 4: 1D Linear Regression for each Predictor
if flag34 == 1
    test3 = zeros(numtarg,numpred);
    test4 = zeros(numtarg,numpred);
    for i = 1:numpred
        pred_cur = pred_arr(:,i);        
        for j = 1:numtarg
            targ_cur = targ_arr(:,j);            
            linregmat = [pred_cur, ones(numdata,1)];
            linregpar = (linregmat'*linregmat)\eye(2)*linregmat'*targ_cur;
            rel_slope = linregpar(1)/linregpar(2);
            test3(j,i) = abs(rel_slope);
            
            data_act = targ_cur;
            data_mod = linregpar(1).*pred_cur+linregpar(2);
            coefdet = 1 - sum((data_act-data_mod).^2) / sum((data_act-mean(data_act)).^2);
            test4(j,i) = coefdet;            
        end
    end
end

test3
test4

%% FS Test 5: Lasso Regression Convergence of Each Parameter
if flag5 == 1
    test5 = zeros(numtarg,numpred);
    for i = 1:numtarg
        [lassomat, fit] = lasso(pred_arr,targ_arr(:,i));
        lassomat = lassomat';
        lambda = fit.Lambda;
        indvec = zeros(1,numpred);
        for j = 1:numpred
            cur_col = lassomat(:,j);
            indvec(j) = find(cur_col,1,'last');
        end
        test5(i,:) = lambda(indvec);
    end
end

test5

%% FS Test 6: Principal Component Analysis
if flag6 == 1
    [wts,~,var] = pca(pred_arr);
    wts = abs(wts(:,1:3));    
    orderval = [];
    curcol = wts(:,1);
    [~,ind] = sort(curcol,'descend');
    orderval = [orderval; ind(1:round(numpred/2))];
    curcol = wts(:,2);
    [~,ind] = sort(curcol,'descend');
    orderval = [orderval; ind(1:round(numpred/2))];
    curcol = wts(:,3);
    [~,ind] = sort(curcol,'descend');
    orderval = [orderval; ind];
    [~, I]=unique(orderval,'first');
    test6 = orderval(sort(I))';
    test6 = repmat(test6,numtarg,1);
end

test6

%% FS Test 7: Number of Times Each Parameter is Included by Seq. Search
if flag7 == 1
    repeats = 1;
    numsim = 50;
    fraction = 0.95;
    test7 = zeros(numtarg,numpred);
    h1 = waitbar(0,'Stepping Through Targets');
    for i = 1:numtarg
        waitbar(i./numtarg,h1);
        cur_class = class2_arr(:,i);
        % Run $repeats of simulations of size $numsims
        summary = zeros(repeats,numpred);
        for j = 1:repeats
            resmat = zeros(numsim,numpred);
            for k = 1:numsim
                randind = randperm(numdata,round(fraction*numdata));
                newpred = pred_arr(randind,:);
                newclass = cur_class(randind,:);
                bestset = plusKminusRc(newpred,newclass);
                resmat(k,bestset) = 1;
            end
            summary(j,:) = sum(resmat,1)./numsim;
        end
        test7(i,:) = mean(summary,1);
    end
end
close all force
test7

save('fs_summary.mat','test1','test2','test3','test4','test5','test6','test7')

[~,order1] = sort(test1,2,'descend');
[~,result1] = sort(order1,2);
[~,order2] = sort(test2,2,'descend');
[~,result2] = sort(order2,2);
[~,order3] = sort(test3,2,'descend');
[~,result3] = sort(order3,2);


test1
result1