function [feature_set, feature_matrix] = plusKminusRc(predictors, targets)

% Implements +3, -2 sequential search for best 5 features, using apparent 
% error of 3NN classifier

% predictors is an array with each column a predictor and observations
% along rows. targets is a column vector with each row an observation.

%% Initialize the Search
numiter = round(size(predictors,2)/2); % Number of iterations to go from empty set to best set of n
forward = 2;
reverse = 1;

bestset = []; % The optimal set of features
addedset = [1:size(predictors,2)]; % The possible features to be added to the set

feature_matrix = zeros(numiter,size([predictors],2));

alldata = [predictors, targets];

for kk = 1:numiter

%% Forward Search Segment
for counts = 1:forward
criter = [];
addedset = [1:size(predictors,2)];
addedset(ismember(addedset,bestset)) = [];
for i = 1:length(addedset)
    curset = sort([bestset,addedset(i)] );        
    curpred = predictors(:,curset);
        
    % 3NN Classifier
    curmdl = fitcknn(curpred, targets, 'NumNeighbors',3);
    
    % Evaluate error
    error = resubLoss(curmdl);      
        
    criter(i) = error;
end
    
bestind = find(criter== min(criter),1);
rand = randperm(length(bestind),1);
bestind = bestind(rand);
bestset = sort([bestset, addedset(bestind)] );
addedset(addedset == addedset(bestind)) = [];
end

%bestset    
%% Reverse Search Segment
minusset = bestset;
for counts = 1:reverse
criter = [];
for i = 1:length(minusset)
    curset = bestset;
    curset(i)=[];
    curset = sort(curset);   
    curpred = predictors(:,curset);
    
    % 3NN Classifier
    curmdl = fitcknn(curpred, targets, 'NumNeighbors',3);
    
    % Evaluate error
    error = resubLoss(curmdl);      
        
    criter(i) = error;
end
    
worstind = find(criter== max(criter),1);
rand = randperm(length(worstind),1);
worstind = worstind(rand);
bestset(worstind) = [];
bestset = sort(bestset);
minusset = bestset;
end

feature_matrix(kk,bestset) = 1;

end

feature_set = bestset;
    
end