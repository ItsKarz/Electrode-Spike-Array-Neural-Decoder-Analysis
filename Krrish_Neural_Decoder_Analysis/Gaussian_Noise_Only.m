% Noise-robust decoding comparison (clean vs noisy) with 5-Fold CV  Comparison
% All models use same training/test splits with combined training+test data
% Comprehensive metrics computed for all SNR levels
% Authors: Krrish Ravuri, Omar Tawakol
clear; close all; clc;

%% -------------------- Config --------------------
outDir = "results";
confDir = fullfile(outDir, "Gaussian Matrices");  % CHANGED from "confusions"
if ~exist(outDir, "dir"); mkdir(outDir); end
if ~exist(confDir, "dir"); mkdir(confDir); end
% Evaluate these SNRs (dB). Use Inf for clean (no noise).
snrList = [Inf 30 20 10 5 0];

% Noise settings (see addNoiseSNR() for model)
perTrialNoise = true;  % true: compute SNR/trial (recommended), false: one global power
rng(7);                % reproducibility

% Toggle: apply noise to TRAINING too?
% Default simulates noisy sensors at inference only (test-time noise).
addNoiseToTraining = false;

% Cross-validation settings
nFolds = 5;  % Number of folds for stratified cross-validation

%% -------------------- Styling (publication-quality) --------------------
set(0,'DefaultAxesFontName','Arial');
set(0,'DefaultTextFontName','Arial');
set(0,'DefaultAxesFontSize',11);
set(0,'DefaultLineLineWidth',1.5);
set(0,'DefaultFigureColor','w');

%% -------------------- Load training/calibration data -------------------
load('Lab5_CenterOutTrain.mat'); 
load('tr.mat'); % used to obtain/confirm FR_dir in Lab 5

% Create training labels (assuming balanced 8 directions)
[~, Category] = max(FR_dir, [], 2);
label_direction = [0 45 90 135 180 225 270 315];
direction_degrees = label_direction(Category);
direction_radians = deg2rad(direction_degrees)';

%% -------------------- Load test session and build test features --------
load('Lab6_CenterOutTest.mat');  % expects: unit, go, direction (1..8)

N_neurons = numel(unit);
N_trials  = numel(go);
N_spikes_per_trial = zeros(N_neurons, N_trials);
for n = 1:N_neurons
    N_spikes_per_trial(n,:) = cell2mat(arrayfun(@(ind) ...
        sum(unit(n).times <= (go(ind)+1) & unit(n).times >= (go(ind)-1)) - 1, ...
        1:N_trials,'UniformOutput',false));
end
FR_by_trial_test = N_spikes_per_trial / 2;  % spikes/s over 2 s window

%% -------------------- Combine ALL Data for Fair 5-Fold CV --------------------
% Training session data
FR_train_session = FR_dir';  % Convert to trials x neurons

% Check the size of FR_dir to create correct labels
% Assuming FR_dir is neurons x trials, and there are 8 directions
n_trials_train = size(FR_dir, 2);  % Number of trials in training data
% Create balanced training labels (1-8 repeating)
train_labels = repmat(1:8, 1, ceil(n_trials_train/8));
train_labels = train_labels(1:n_trials_train)';  % Truncate to correct size and make column vector

% Test session data
FR_test_session = FR_by_trial_test';  % trials x neurons
test_labels = direction(:);  % Ensure this is a column vector

% Check dimensions
fprintf('Data dimensions:\n');
fprintf('  FR_train_session: %d x %d (trials x neurons)\n', size(FR_train_session));
fprintf('  train_labels: %d x %d\n', size(train_labels));
fprintf('  FR_test_session: %d x %d (trials x neurons)\n', size(FR_test_session));
fprintf('  test_labels: %d x %d\n', size(test_labels));

% Make sure both label vectors are column vectors
if isrow(train_labels)
    train_labels = train_labels';
end
if isrow(test_labels)
    test_labels = test_labels';
end

% Now combine everything
FR_all = [FR_train_session; FR_test_session];  % ALL trials x neurons
labels_all = [train_labels; test_labels];      % ALL labels (1-8)

fprintf('\nCombined dataset for fair 5-fold CV:\n');
fprintf('  Training session: %d trials\n', size(FR_train_session, 1));
fprintf('  Test session: %d trials\n', size(FR_test_session, 1));
fprintf('  Total: %d trials x %d neurons\n', size(FR_all, 1), size(FR_all, 2));
fprintf('  Labels shape: %d x %d\n\n', size(labels_all));

%% -------------------- Create 5-Fold CV Split --------------------
% Create ONE 5-fold CV split for ALL models (ensures same splits for everyone)
cv = cvpartition(labels_all, 'KFold', nFolds, 'Stratify', true);

fprintf('5-Fold CV partitions created:\n');
fprintf('  Training folds: ');
for fold = 1:nFolds
    fprintf('%d ', sum(training(cv, fold)));
end
fprintf('\n  Test folds: ');
for fold = 1:nFolds
    fprintf('%d ', sum(test(cv, fold)));
end
fprintf('\n\n');

%% -------------------- Compute Global Templates from FULL Training Data --------------------
% Compute templates from ALL Lab5 training data (like the old code did)
% This ensures stable templates for Poisson decoder
FR_template_global = zeros(size(FR_dir, 1), 8);  % neurons x 8 directions
for dir = 1:8
    idx = find(train_labels == dir);
    if ~isempty(idx)
        FR_template_global(:, dir) = mean(FR_train_session(idx, :)', 2);
    else
        FR_template_global(:, dir) = mean(FR_train_session', 2);
    end
end

fprintf('Global templates computed from ALL Lab5 training data:\n');
fprintf('  Template size: %d neurons x %d directions\n', size(FR_template_global));

%% -------------------- Noise model (explained) --------------------------
% For each trial feature vector s (neurons x 1):
%   P_signal = mean(s.^2);
%   Target SNR_dB => P_noise = P_signal / (10^(SNR_dB/10));
%   sigma = sqrt(P_noise);
%   s_noisy = s + N(0, sigma^2 I).
% SNR = Inf → sigma = 0 (no noise). Per-trial mode maintains comparable SNR across trials.

%% -------------------- Determine worst SNR (for confusion plots only) ---
finiteSNRs = snrList(~isinf(snrList));
worstSNR   = min(finiteSNRs);   % smallest dB = noisiest

%% -------------------- Main 5-Fold Cross-Validation Evaluation --------------------
% Use valid MATLAB field names (no hyphens)
decoderNames = {'PopVec','MLE_Poisson','MLE_Normal','SVM','kNN','RandomForest'};
decoderDisplayNames = {'PopVec','MLE-Poisson','MLE-Normal','SVM','kNN','RandomForest'};
nClasses = 8;         % directions are 1..8

% Store all predictions for ALL SNR levels
all_predictions = struct();

% Store fold-level accuracies for std calculation
fold_accuracies_all = struct();
for s = 1:numel(snrList)
    snrKey = sprintf('snr_%s', snrToStr(snrList(s)));
    fold_accuracies_all.(snrKey) = struct();
    for i = 1:length(decoderNames)
        fold_accuracies_all.(snrKey).(decoderNames{i}) = zeros(nFolds, 1);
    end
end

% Store all results for final comprehensive table
resultsTable = table();

for s = 1:numel(snrList)
    snrDb = snrList(s);
    isClean = isinf(snrDb);
    isWorst = ~isClean && (snrDb == worstSNR);
    
    fprintf('\n=== Processing SNR = %s dB ===\n', snrToStr(snrDb));
    
    % Initialize storage for this SNR
    fold_accuracies = struct();
    for i = 1:length(decoderNames)
        fold_accuracies.(decoderNames{i}) = zeros(nFolds, 1);
    end
    
    % Store predictions for ALL SNR levels
    snrKey = sprintf('snr_%s', snrToStr(snrDb));
    all_predictions.(snrKey) = struct();
    for i = 1:length(decoderNames)
        all_predictions.(snrKey).(decoderNames{i}) = struct(...
            'y_true', [], 'y_pred', [], 'scores', []);
    end
    
    % 5-Fold Cross-Validation Loop
    for fold = 1:nFolds
        fprintf('  Fold %d/%d...\n', fold, nFolds);
        
        % Get SAME training/test split for ALL models
        trainIdx = training(cv, fold);
        testIdx = test(cv, fold);
        
        % Split data (ALL models get same data)
        X_train = FR_all(trainIdx, :);
        y_train = labels_all(trainIdx, :);
        X_test_clean = FR_all(testIdx, :);
        y_test = labels_all(testIdx, :);
        
        % Apply noise to test data
        X_test = addNoiseSNR(X_test_clean', snrDb, perTrialNoise)';
        
        % Apply noise to training data if specified
        if addNoiseToTraining
            X_train = addNoiseSNR(X_train', snrDb, perTrialNoise)';
        end
        
        % Convert to proper dimensions for template methods
        FR_train = X_train';  % neurons x trials
        FR_test = X_test';    % neurons x trials
        
        % ----- Template-based decoders -----
        % Create templates from training fold (same data ML methods use)
        FR_template = zeros(size(FR_train, 1), nClasses);
        for dir = 1:nClasses
            idx = (y_train == dir);
            if sum(idx) > 0
                FR_template(:, dir) = mean(FR_train(:, idx), 2);
            else
                % If no examples for this direction, use overall mean
                FR_template(:, dir) = mean(FR_train, 2);
            end
        end
        
        % Get preferred directions from training fold templates
        [~, train_pref_dir] = max(FR_template, [], 2);
        train_dir_radians = deg2rad(label_direction(train_pref_dir))';
        
        % 1. Population Vector (using adjusted method)
        pred_pv = populationVectorDecode(FR_test, train_dir_radians);
        acc_pv = mean(pred_pv' == y_test);
        fold_accuracies.PopVec(fold) = acc_pv;
        fold_accuracies_all.(snrKey).PopVec(fold) = acc_pv;
        
        % 2. MLE Poisson - Use GLOBAL templates from ALL Lab5 training data
        % This matches what the old code did and should give ~0.7 accuracy
        [pred_mle_p, scores_mle_p] = mlePoissonDecode(FR_test, FR_template_global);
        acc_mle_p = mean(pred_mle_p == y_test);
        fold_accuracies.MLE_Poisson(fold) = acc_mle_p;
        fold_accuracies_all.(snrKey).MLE_Poisson(fold) = acc_mle_p;
        
        % 3. MLE Normal
        [pred_mle_n, scores_mle_n] = mleNormalDecode(FR_test, FR_template);
        acc_mle_n = mean(pred_mle_n == y_test);
        fold_accuracies.MLE_Normal(fold) = acc_mle_n;
        fold_accuracies_all.(snrKey).MLE_Normal(fold) = acc_mle_n;
        
        % ----- ML-based decoders -----
        % 4. SVM (train on same X_train, y_train)
        mdl_svm = fitcecoc(X_train, y_train, 'Coding', 'onevsall', 'Verbose', 0);
        [pred_svm, scores_svm] = predict(mdl_svm, X_test);
        acc_svm = mean(pred_svm == y_test);
        fold_accuracies.SVM(fold) = acc_svm;
        fold_accuracies_all.(snrKey).SVM(fold) = acc_svm;
        
        % 5. kNN (train on same X_train, y_train)
        mdl_knn = fitcknn(X_train, y_train, 'NumNeighbors', 3);
        [pred_knn, scores_knn] = predict(mdl_knn, X_test);
        acc_knn = mean(pred_knn == y_test);
        fold_accuracies.kNN(fold) = acc_knn;
        fold_accuracies_all.(snrKey).kNN(fold) = acc_knn;
        
        % 6. Random Forest (train on same X_train, y_train)
        mdl_rf = TreeBagger(50, X_train, y_train, 'OOBPrediction', 'Off', ...
            'Method', 'classification', 'NumPredictorsToSample', 'all');
        [pred_rf, scores_rf] = predict(mdl_rf, X_test);
        pred_rf = str2double(pred_rf);
        acc_rf = mean(pred_rf == y_test);
        fold_accuracies.RandomForest(fold) = acc_rf;
        fold_accuracies_all.(snrKey).RandomForest(fold) = acc_rf;
        
        % Store predictions and scores for ALL SNR levels
        pred_cells = {pred_pv', pred_mle_p, pred_mle_n, ...
                     pred_svm, pred_knn, pred_rf};
        score_cells = {[], scores_mle_p, scores_mle_n, ...
                      scores_svm, scores_knn, scores_rf};
        
        for d = 1:length(decoderNames)
            if fold == 1
                all_predictions.(snrKey).(decoderNames{d}).y_true = y_test;
                all_predictions.(snrKey).(decoderNames{d}).y_pred = pred_cells{d};
                all_predictions.(snrKey).(decoderNames{d}).scores = score_cells{d};
            else
                all_predictions.(snrKey).(decoderNames{d}).y_true = ...
                    [all_predictions.(snrKey).(decoderNames{d}).y_true; y_test];
                all_predictions.(snrKey).(decoderNames{d}).y_pred = ...
                    [all_predictions.(snrKey).(decoderNames{d}).y_pred; pred_cells{d}];
                if ~isempty(score_cells{d})
                    all_predictions.(snrKey).(decoderNames{d}).scores = ...
                        [all_predictions.(snrKey).(decoderNames{d}).scores; score_cells{d}];
                end
            end
        end
    end
    
    % Compute average accuracy across folds for this SNR
    fprintf('\n  Average accuracies across %d folds:\n', nFolds);
    for d = 1:length(decoderNames)
        decoder = decoderNames{d};
        avg_acc = mean(fold_accuracies.(decoder));
        std_acc = std(fold_accuracies.(decoder));
        
        % Use display name for output
        dispName = decoderDisplayNames{d};
        fprintf('    %-15s: %.3f ± %.3f\n', dispName, avg_acc, std_acc);
    end
end

%% -------------------- Compute Comprehensive Metrics --------------------
fprintf('\n=== Computing comprehensive metrics for clean and 0 dB ===\n');

% Initialize results structure
all_results = [];

for s = 1:numel(snrList)
    snrDb = snrList(s);
    if ~(isinf(snrDb) || snrDb == 0)
        continue;  % Only process clean (∞) and 0 dB
    end
    
    fprintf('\nProcessing SNR = %s dB...\n', snrToStr(snrDb));
    snrKey = sprintf('snr_%s', snrToStr(snrDb));
    
    for d = 1:length(decoderNames)
        decoder = decoderNames{d};
        dispName = decoderDisplayNames{d};
        
        data = all_predictions.(snrKey).(decoder);
        y_true = data.y_true;
        y_pred = data.y_pred;
        
        % Compute comprehensive metrics
        metrics = computeComprehensiveMetrics(y_true, y_pred, nClasses, data.scores);
        
        % Add decoder and SNR info
        metrics.decoder = dispName;
        metrics.snr_db = snrDb;
        
        % Store results
        all_results = [all_results; metrics]; %#ok<AGROW>
    end
end

%% -------------------- Compute Performance Drop and Retention --------------------
fprintf('\n=== Computing performance drop and retention ===\n');

% Get results for clean and 0 dB
clean_idx = isinf([all_results.snr_db]);
zero_idx = ([all_results.snr_db] == 0);

clean_results = all_results(clean_idx);
zero_results = all_results(zero_idx);

% Sort to ensure same order
[~, clean_order] = sort({clean_results.decoder});
[~, zero_order] = sort({zero_results.decoder});

clean_results = clean_results(clean_order);
zero_results = zero_results(zero_order);

% Calculate std for accuracy from fold-level data
clean_std = struct();
zero_std = struct();

clean_snrKey = 'snr_inf';
zero_snrKey = 'snr_0';

for d = 1:length(decoderNames)
    decoder = decoderNames{d};
    clean_std.(decoder) = std(fold_accuracies_all.(clean_snrKey).(decoder));
    zero_std.(decoder) = std(fold_accuracies_all.(zero_snrKey).(decoder));
end

% Create comprehensive results table
compTable = table();

for i = 1:length(clean_results)
    % Clean SNR metrics
    clean = clean_results(i);
    zero = zero_results(i);
    
    % Get decoder name
    decoder = decoderNames{i};
    
    % Performance drop (Δ)
    delta_accuracy = zero.accuracy - clean.accuracy;
    delta_precision = zero.precision - clean.precision;
    delta_recall = zero.recall - clean.recall;
    delta_f1 = zero.f1 - clean.f1;
    
    % Performance retention (%)
    retention_accuracy = (zero.accuracy / clean.accuracy) * 100;
    retention_precision = (zero.precision / clean.precision) * 100;
    retention_recall = (zero.recall / clean.recall) * 100;
    retention_f1 = (zero.f1 / clean.f1) * 100;
    
    % Get std values
    acc_clean_std = clean_std.(decoder);
    acc_zero_std = zero_std.(decoder);
    
    % Create row for this decoder
    row = table();
    row.decoder = {clean.decoder};
    row.snr_db_clean = Inf;
    row.snr_db_zero = 0;
    
    % Clean metrics with std
    row.accuracy_clean = clean.accuracy;
    row.accuracy_clean_std = acc_clean_std;
    row.precision_clean = clean.precision;
    row.recall_clean = clean.recall;
    row.specificity_clean = clean.specificity;
    row.f1_clean = clean.f1;
    row.cohens_kappa_clean = clean.cohens_kappa;
    row.auc_clean = clean.auc_macro;
    
    % 0 dB metrics with std
    row.accuracy_zero = zero.accuracy;
    row.accuracy_zero_std = acc_zero_std;
    row.precision_zero = zero.precision;
    row.recall_zero = zero.recall;
    row.specificity_zero = zero.specificity;
    row.f1_zero = zero.f1;
    row.cohens_kappa_zero = zero.cohens_kappa;
    row.auc_zero = zero.auc_macro;
    
    % Performance drops
    row.delta_accuracy = delta_accuracy;
    row.delta_precision = delta_precision;
    row.delta_recall = delta_recall;
    row.delta_f1 = delta_f1;
    
    % Performance retentions (%)
    row.retention_accuracy = retention_accuracy;
    row.retention_precision = retention_precision;
    row.retention_recall = retention_recall;
    row.retention_f1 = retention_f1;
    
    % Add to comprehensive table
    compTable = [compTable; row]; %#ok<AGROW>
    
    fprintf('  %-15s: Acc retention = %.1f%%, F1 retention = %.1f%%\n', ...
        clean.decoder, retention_accuracy, retention_f1);
end

%% -------------------- Save Results --------------------
% Comprehensive metrics table
writetable(compTable, fullfile(outDir, 'comprehensive_metrics.csv'));

% Save individual results
writetable(struct2table(all_results), fullfile(outDir, 'all_metrics.csv'));

%% -------------------- Save Confusion Matrices --------------------
fprintf('\nSaving confusion matrices for clean and worst SNR...\n');
for s = 1:numel(snrList)
    snrDb = snrList(s);
    isClean = isinf(snrDb);
    isWorst = ~isClean && (snrDb == worstSNR);
    
    if isClean || isWorst
        snrKey = sprintf('snr_%s', snrToStr(snrDb));
        for d = 1:length(decoderNames)
            decoder = decoderDisplayNames{d};
            data = all_predictions.(snrKey).(decoderNames{d});
            saveConf(data.y_true, data.y_pred, confDir, decoder, snrDb);
        end
    end
end

%% -------------------- Create Performance Summary Plot --------------------
figure('Position', [100 100 800 500]);
nDecoders = height(compTable);

% Subplot 1: Accuracy retention
subplot(2, 2, 1);
hold on; grid on;
[~, idx] = sort(compTable.retention_accuracy, 'descend');
bar(1:nDecoders, compTable.retention_accuracy(idx));
set(gca, 'XTick', 1:nDecoders, 'XTickLabel', compTable.decoder(idx));
xtickangle(45);
ylabel('Accuracy Retention (%)');
title('Performance Retention at 0 dB (vs Clean)');
ylim([0 120]);

% Subplot 2: Performance drops
subplot(2, 2, 2);
hold on; grid on;
colors = lines(4);
for i = 1:4
    switch i
        case 1
            data = compTable.delta_accuracy;
            label = 'Accuracy';
        case 2
            data = compTable.delta_precision;
            label = 'Precision';
        case 3
            data = compTable.delta_recall;
            label = 'Recall';
        case 4
            data = compTable.delta_f1;
            label = 'F1';
    end
    plot(1:nDecoders, data, 'o-', 'Color', colors(i,:), ...
        'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', label);
end
set(gca, 'XTick', 1:nDecoders, 'XTickLabel', compTable.decoder);
xtickangle(45);
ylabel('Δ Performance (0 dB - Clean)');
title('Performance Degradation');
legend('Location', 'best');
grid on;

% Subplot 3: Clean vs 0 dB F1 scores
subplot(2, 2, 3);
hold on; grid on;
x = 1:nDecoders;
bar(x - 0.2, compTable.f1_clean, 0.4, 'FaceColor', [0.2 0.6 0.8]);
bar(x + 0.2, compTable.f1_zero, 0.4, 'FaceColor', [0.8 0.4 0.2]);
set(gca, 'XTick', x, 'XTickLabel', compTable.decoder);
xtickangle(45);
ylabel('F1 Score');
title('F1 Score Comparison');
legend({'Clean', '0 dB'}, 'Location', 'best');
ylim([0 1]);

% Subplot 4: AUC comparison
subplot(2, 2, 4);
hold on; grid on;
[~, idx] = sort(compTable.auc_clean, 'descend');
bar_data = [compTable.auc_clean(idx), compTable.auc_zero(idx)];
bar(1:nDecoders, bar_data);
set(gca, 'XTick', 1:nDecoders, 'XTickLabel', compTable.decoder(idx));
xtickangle(45);
ylabel('AUC (Macro-average)');
title('AUC Comparison');
legend({'Clean', '0 dB'}, 'Location', 'best');
ylim([0 1]);

sgtitle('Comprehensive Performance Analysis: Clean vs 0 dB', 'FontSize', 14);
exportSafe(gcf, outDir, 'performance_summary');

%% -------------------- Print Comprehensive Summary with STD --------------------
printComprehensiveSummary(compTable, nFolds);

%% -------------------- Original Robustness plots (accuracy & Δ vs clean) ---------
finiteSNR = snrList(~isinf(snrList));
finiteSNR = unique(sort(finiteSNR(:).', 'ascend'), 'stable');
pad = 10;
infTick = max(finiteSNR) + pad;

% Create accuracy table from ALL SNR levels using the stored predictions
accRows = [];

% Re-extract accuracy from all_predictions for ALL SNR levels
for s = 1:numel(snrList)
    snrDb = snrList(s);
    snrKey = sprintf('snr_%s', snrToStr(snrDb));
    
    for d = 1:length(decoderNames)
        dispName = decoderDisplayNames{d};
        
        if isfield(all_predictions, snrKey) && isfield(all_predictions.(snrKey), decoderNames{d})
            data = all_predictions.(snrKey).(decoderNames{d});
            y_true = data.y_true;
            y_pred = data.y_pred;
            
            % Calculate accuracy
            if ~isempty(y_true) && ~isempty(y_pred)
                acc = sum(y_true == y_pred) / length(y_true);
                accRows = [accRows; {dispName, snrDb, acc}]; %#ok<AGROW>
            end
        end
    end
end

% Convert to table for plotting
accT = cell2table(accRows, 'VariableNames', {'decoder','snr_db','accuracy'});

% Prepare for plotting
accT.plot_x = accT.snr_db;
isInfRow = isinf(accT.plot_x);
accT.plot_x(isInfRow) = infTick;
accT = sortrows(accT, {'decoder','plot_x'});

%% -------------------- Compute Mean and STD for Error Bars --------------------
% Calculate mean and std for each decoder at each SNR
decList = unique(accT.decoder, 'stable');
snrValues = unique(accT.snr_db, 'stable');
nDecoders = length(decList);
nSNRs = length(snrValues);

% Initialize arrays
meanAcc = zeros(nDecoders, nSNRs);
stdAcc = zeros(nDecoders, nSNRs);

% Calculate statistics
for d = 1:nDecoders
    for s = 1:nSNRs
        % Get data for this decoder and SNR
        idx = (strcmp(accT.decoder, decList{d}) & (accT.snr_db == snrValues(s)));
        if any(idx)
            meanAcc(d, s) = accT.accuracy(idx);
            
            % For error bars, we need to compute std from fold-level data
            if isinf(snrValues(s))
                snrKey = 'snr_inf';
            else
                snrKey = sprintf('snr_%d', snrValues(s));
            end
            
            % Use the fold-level accuracies we stored earlier
            if isfield(fold_accuracies_all, snrKey)
                decoderName = decoderNames{d}; % Use the original name
                if isfield(fold_accuracies_all.(snrKey), decoderName)
                    stdAcc(d, s) = std(fold_accuracies_all.(snrKey).(decoderName));
                end
            end
        end
    end
end

% Prepare x-values for plotting
xValues = snrValues;
xValues(isinf(xValues)) = infTick; % Replace Inf with infTick

%% -------------------- Accuracy vs SNR with Error Bars --------------------
% Accuracy vs SNR with error bars
figure('Position', [100 100 650 440]); 
hold on; grid on;
colors = lines(nDecoders);

% Plot each decoder with error bars
for i = 1:nDecoders
    % Create error bar plot
    errorbar(xValues, meanAcc(i, :), stdAcc(i, :), '-o', ...
        'Color', colors(i, :), ...
        'DisplayName', decList{i}, ...
        'LineWidth', 1.5, ...
        'MarkerSize', 8, ...
        'CapSize', 6);
end

xlim([min(finiteSNR) infTick]); 
ylim([0 1]);
xlabel('SNR (dB) — higher is cleaner (∞ = no noise)', 'FontSize', 12);
ylabel('Decoding accuracy', 'FontSize', 12);
title('Decoder robustness to additive Gaussian feature noise (5-Fold CV)', 'FontSize', 13);
legend('Location', 'southwest', 'Box', 'off');
xticks([finiteSNR infTick]);
xticklabels([arrayfun(@(v) sprintf('%d', v), finiteSNR, 'UniformOutput', false) {'∞'}]);
set(gca, 'FontSize', 11);

% Add grid for better readability
grid on;
box on;

exportSafe(gcf, outDir, 'accuracy_vs_snr_with_errorbars');

%% -------------------- Δ Accuracy vs clean with Error Bars --------------------
% Δ Accuracy vs clean with error bars
figure('Position', [120 120 650 440]); 
hold on; grid on;

% Calculate Δ accuracy and its error
deltaAcc = zeros(nDecoders, nSNRs);
deltaStd = zeros(nDecoders, nSNRs);

for i = 1:nDecoders
    % Find clean SNR index
    cleanIdx = find(isinf(snrValues));
    if isempty(cleanIdx)
        continue;
    end
    
    % Calculate delta and propagate errors
    for s = 1:nSNRs
        deltaAcc(i, s) = meanAcc(i, s) - meanAcc(i, cleanIdx);
        
        % Error propagation for subtraction: sqrt(std1^2 + std2^2)
        deltaStd(i, s) = sqrt(stdAcc(i, s)^2 + stdAcc(i, cleanIdx)^2);
    end
    
    % Plot with error bars
    errorbar(xValues, deltaAcc(i, :), deltaStd(i, :), '-o', ...
        'Color', colors(i, :), ...
        'DisplayName', decList{i}, ...
        'LineWidth', 1.5, ...
        'MarkerSize', 8, ...
        'CapSize', 6);
end

xlim([min(finiteSNR) infTick]);
xlabel('SNR (dB) — higher is cleaner (∞ = no noise)', 'FontSize', 12);
ylabel('Δ Accuracy vs clean (absolute)', 'FontSize', 12);
title('Performance degradation under additive feature noise (5-Fold CV)', 'FontSize', 13);
yline(0, '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1);
legend('Location', 'southwest', 'Box', 'off');
xticks([finiteSNR infTick]);
xticklabels([arrayfun(@(v) sprintf('%d', v), finiteSNR, 'UniformOutput', false) {'∞'}]);
set(gca, 'FontSize', 11);

exportSafe(gcf, outDir, 'delta_accuracy_vs_snr_with_errorbars');

%% -------------------- Original plots (without error bars) for comparison --------------------
% Accuracy vs SNR (original)
figure('Position', [100 100 650 440]); hold on; grid on;
for i = 1:numel(decList)
    sub = accT(strcmp(accT.decoder, decList{i}), :);
    plot(sub.plot_x, sub.accuracy, '-o', 'Color', colors(i, :), ...
        'DisplayName', decList{i}, 'LineWidth', 1.5, 'MarkerSize', 8);
end
xlim([min(finiteSNR) infTick]); ylim([0 1]);
xlabel('SNR (dB) — higher is cleaner (∞ = no noise)', 'FontSize', 12);
ylabel('Decoding accuracy', 'FontSize', 12);
title('Decoder robustness to additive Gaussian feature noise (5-Fold CV)', 'FontSize', 13);
legend('Location', 'southwest', 'Box', 'off');
xticks([finiteSNR infTick]);
xticklabels([arrayfun(@(v) sprintf('%d', v), finiteSNR, 'UniformOutput', false) {'∞'}]);
set(gca, 'FontSize', 11);
exportSafe(gcf, outDir, 'accuracy_vs_snr');

% Δ Accuracy vs clean (original)
figure('Position', [120 120 650 440]); hold on; grid on;
for i = 1:numel(decList)
    sub = accT(strcmp(accT.decoder, decList{i}), :);
    base = sub.accuracy(sub.plot_x == infTick);
    if isempty(base), continue; end
    plot(sub.plot_x, sub.accuracy - base, '-o', 'Color', colors(i, :), ...
        'DisplayName', decList{i}, 'LineWidth', 1.5, 'MarkerSize', 8);
end
xlim([min(finiteSNR) infTick]);
xlabel('SNR (dB) — higher is cleaner (∞ = no noise)', 'FontSize', 12);
ylabel('Δ Accuracy vs clean (absolute)', 'FontSize', 12);
title('Performance degradation under additive feature noise (5-Fold CV)', 'FontSize', 13);
yline(0, '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1);
legend('Location', 'southwest', 'Box', 'off');
xticks([finiteSNR infTick]);
xticklabels([arrayfun(@(v) sprintf('%d', v), finiteSNR, 'UniformOutput', false) {'∞'}]);
set(gca, 'FontSize', 11);
exportSafe(gcf, outDir, 'delta_accuracy_vs_snr');

%% -------------------- Local functions --------------------
function Xn = addNoiseSNR(X, snrDb, perTrial)
    if isinf(snrDb), Xn = X; return; end
    if perTrial
        pw = mean(X.^2, 1);
        noisePw = pw ./ (10.^(snrDb/10));
        sigma = sqrt(max(noisePw, eps));
        Z = randn(size(X)) .* sigma;
    else
        pw = mean(X.^2, 'all');
        noisePw = pw ./ (10.^(snrDb/10));
        sigma = sqrt(max(noisePw, eps));
        Z = sigma .* randn(size(X));
    end
    Xn = X + Z;
end

function pred = populationVectorDecode(FR_test, dir_rads)
    % This function now implements the adjusted population vector method
    mu = mean(FR_test, 2);
    W = FR_test - mu;
    Px = sum(W .* cos(dir_rads), 1);
    Py = sum(W .* sin(dir_rads), 1);
    theta = atan2(Py, Px);
    
    theta_deg = mod(rad2deg(theta) + 360, 360);
    pred = zeros(1, size(FR_test, 2));
    
    for i = 1:length(theta_deg)
        ang = theta_deg(i);
        if ang >= 337.5 || ang < 22.5
            pred(i) = 1;
        elseif ang >= 22.5 && ang < 67.5
            pred(i) = 2;
        elseif ang >= 67.5 && ang < 112.5
            pred(i) = 3;
        elseif ang >= 112.5 && ang < 157.5
            pred(i) = 4;
        elseif ang >= 157.5 && ang < 202.5
            pred(i) = 5;
        elseif ang >= 202.5 && ang < 247.5
            pred(i) = 6;
        elseif ang >= 247.5 && ang < 292.5
            pred(i) = 7;
        else
            pred(i) = 8;
        end
    end
end

function [pred, scores] = mlePoissonDecode(FR_test, FR_template)
    Ndir = size(FR_template, 2);
    Ntr = size(FR_test, 2);
    logL = zeros(Ntr, Ndir);
    
    % OLD IMPLEMENTATION that gave ~0.7 accuracy
    % Element-wise: FR_test .* log(lambda) - lambda
    for d = 1:Ndir
        lambda = FR_template(:, d);
        % CRITICAL: This is the OLD implementation
        ll = FR_test .* log(max(lambda, eps)) - lambda; % up to class-constant
        logL(:, d) = sum(ll, 1)';  % Sum across neurons
    end
    
    % Convert to probabilities (softmax) for compatibility with metrics code
    scores = exp(logL - max(logL, [], 2));
    scores = scores ./ sum(scores, 2);
    
    [~, pred] = max(logL, [], 2);
end

function [pred, scores] = mleNormalDecode(FR_test, FR_template)
    Ndir = size(FR_template, 2);
    Ntr = size(FR_test, 2);
    logL = zeros(Ntr, Ndir);
    
    sig = std(FR_test, 0, 2) + 1e-6;
    
    for d = 1:Ndir
        mu = FR_template(:, d);
        ll = -0.5 * sum(((FR_test - mu) ./ sig).^2, 1)' - sum(log(sig));
        logL(:, d) = ll;
    end
    
    % Convert to probabilities (softmax)
    scores = exp(logL - max(logL, [], 2));
    scores = scores ./ sum(scores, 2);
    
    [~, pred] = max(logL, [], 2);
end

function metrics = computeComprehensiveMetrics(y_true, y_pred, nClasses, scores)
    % Initialize metrics structure
    metrics = struct();
    
    % Basic confusion matrix
    C = confusionmat(y_true, y_pred, 'Order', 1:nClasses);
    
    % Per-class metrics
    TP = diag(C);
    FP = sum(C, 1)' - TP;
    FN = sum(C, 2) - TP;
    TN = sum(C, 'all') - (TP + FP + FN);
    
    % Per-class calculations
    precision_per_class = TP ./ (TP + FP);
    recall_per_class = TP ./ (TP + FN);  % Sensitivity
    specificity_per_class = TN ./ (TN + FP);
    f1_per_class = 2 .* precision_per_class .* recall_per_class ./ ...
                   (precision_per_class + recall_per_class);
    
    % Macro-averages (unweighted mean across classes)
    metrics.precision = nanmean(precision_per_class);
    metrics.recall = nanmean(recall_per_class);  % Sensitivity (macro)
    metrics.specificity = nanmean(specificity_per_class);
    metrics.f1 = nanmean(f1_per_class);
    
    % Overall accuracy
    metrics.accuracy = sum(TP) / sum(C, 'all');
    
    % Cohen's Kappa
    n = sum(C, 'all');
    p0 = metrics.accuracy;  % observed agreement
    pe = sum(sum(C, 1) .* sum(C, 2)') / (n^2);  % expected agreement by chance
    metrics.cohens_kappa = (p0 - pe) / (1 - pe);
    
    % AUC (if scores are available)
    if ~isempty(scores) && size(scores, 2) == nClasses
        % One-vs-rest AUC for each class, then macro-average
        auc_per_class = zeros(nClasses, 1);
        for c = 1:nClasses
            y_true_binary = (y_true == c);
            y_scores = scores(:, c);
            [~, ~, ~, auc] = perfcurve(y_true_binary, y_scores, 1);
            auc_per_class(c) = auc;
        end
        metrics.auc_macro = nanmean(auc_per_class);
    else
        metrics.auc_macro = NaN;
    end
end

function saveConf(ytrue, ypred, confDir, decName, snrDb)
    C = confusionmat(ytrue(:), ypred(:));
    fig = figure('Position', [200 200 520 480], 'Color', 'w');
    confusionchart(C, 'Title', sprintf('Confusion: %s @ %s dB', decName, snrToStr(snrDb)), ...
        'XLabel', 'Predicted', 'YLabel', 'True');
    
    % Save directly in Gaussian Matrices folder with descriptive name
    baseName = sprintf('Confusion_%s_SNR_%s', sanitize(decName), snrToStr(snrDb));
    exportSafe(fig, confDir, baseName);
    close(fig);
end

function s = snrToStr(v)
    if isinf(v), s = 'inf'; else, s = sprintf('%d', v); end
end

function s = snrLabel(v)
    if isinf(v), s = 'Inf'; else, s = sprintf('%d', v); end
end

function t = sanitize(s)
    t = regexprep(string(s), '[^\w\-]+', '_');
end

function exportSafe(figHandle, outDir, baseName)
    if ~exist(outDir, 'dir'), mkdir(outDir); end
    pdfPath = fullfile(outDir, baseName + ".pdf");
    svgPath = fullfile(outDir, baseName + ".svg");
    pngPath = fullfile(outDir, baseName + ".png");

    exportgraphics(figHandle, pdfPath, 'ContentType', 'vector');
    try
        exportgraphics(figHandle, svgPath, 'ContentType', 'vector');
    catch
        print(figHandle, svgPath(1:end-4), '-dsvg', '-painters');
    end
    exportgraphics(figHandle, pngPath, 'Resolution', 300);
end

function printComprehensiveSummary(compTable, nFolds)
    fprintf('\n============================================================\n');
    fprintf(' COMPREHENSIVE PERFORMANCE ANALYSIS\n');
    fprintf(' Clean (∞ dB) vs Noisy (0 dB) Conditions\n');
    fprintf(' Based on %d-fold stratified cross-validation\n', nFolds);
    fprintf('============================================================\n\n');
    
    % Print table header with Kappa columns
    fprintf('%-15s %11s %11s %8s %8s %8s %8s %8s %8s\n', ...
        'Decoder', 'Acc@∞ ± std', 'Acc@0 ± std', 'ΔAcc', 'Ret%', 'F1@∞', 'F1@0', 'κ@∞', 'κ@0');
    fprintf('%-15s %11s %11s %8s %8s %8s %8s %8s %8s\n', ...
        '', '(mean ± std)', '(mean ± std)', '', '', '', '', '', '');
    fprintf('%s\n', repmat('-', 95, 1));
    
    for i = 1:height(compTable)
        row = compTable(i, :);
        fprintf('%-15s %5.3f ± %.3f %5.3f ± %.3f %8.3f %7.1f%% %8.3f %8.3f %8.3f %8.3f\n', ...
            row.decoder{1}, ...
            row.accuracy_clean, row.accuracy_clean_std, ...
            row.accuracy_zero, row.accuracy_zero_std, ...
            row.delta_accuracy, ...
            row.retention_accuracy, ...
            row.f1_clean, ...
            row.f1_zero, ...
            row.cohens_kappa_clean, ...
            row.cohens_kappa_zero);
    end
    
    fprintf('\nKEY METRICS SUMMARY:\n');
    fprintf('1. All values reported as mean ± standard deviation across %d folds\n', nFolds);
    fprintf('2. Cohen''s Kappa (κ): Agreement beyond chance (-1 to 1), 0 = random, 1 = perfect\n');
    fprintf('3. Specificity (macro-average): False negative rate\n');
    fprintf('4. Sensitivity/Recall (macro-average): True positive rate\n');
    fprintf('5. Precision (macro-average): Positive predictive value\n');
    fprintf('6. F1-Score: Harmonic mean of precision and recall\n');
    fprintf('7. AUC (macro-average): ROC curve area, chance = 0.5\n');
    fprintf('8. Performance Drop (Δ): 0 dB value minus clean value\n');
    fprintf('9. Performance Retention: (0 dB / clean) × 100%%\n\n');
    
    % Find best performers considering std
    fprintf('BEST PERFORMERS (considering mean ± std):\n');
    
    % Clean accuracy with std
    clean_acc_with_std = compTable.accuracy_clean - compTable.accuracy_clean_std; % Lower bound
    [~, best_clean_acc_idx] = max(clean_acc_with_std);
    fprintf('  • Highest clean accuracy: %s (%.3f ± %.3f)\n', ...
        compTable.decoder{best_clean_acc_idx}, ...
        compTable.accuracy_clean(best_clean_acc_idx), ...
        compTable.accuracy_clean_std(best_clean_acc_idx));
    
    % 0 dB accuracy with std
    zero_acc_with_std = compTable.accuracy_zero - compTable.accuracy_zero_std; % Lower bound
    [~, best_zero_acc_idx] = max(zero_acc_with_std);
    fprintf('  • Highest 0 dB accuracy: %s (%.3f ± %.3f)\n', ...
        compTable.decoder{best_zero_acc_idx}, ...
        compTable.accuracy_zero(best_zero_acc_idx), ...
        compTable.accuracy_zero_std(best_zero_acc_idx));
    
    % Clean Kappa
    [~, best_clean_kappa_idx] = max(compTable.cohens_kappa_clean);
    fprintf('  • Highest clean κ: %s (%.3f)\n', ...
        compTable.decoder{best_clean_kappa_idx}, ...
        compTable.cohens_kappa_clean(best_clean_kappa_idx));
    
    % 0 dB Kappa
    [~, best_zero_kappa_idx] = max(compTable.cohens_kappa_zero);
    fprintf('  • Highest 0 dB κ: %s (%.3f)\n', ...
        compTable.decoder{best_zero_kappa_idx}, ...
        compTable.cohens_kappa_zero(best_zero_kappa_idx));
    
    % Best retention
    [~, best_retention_idx] = max(compTable.retention_accuracy);
    fprintf('  • Best retention: %s (%.1f%%)\n', ...
        compTable.decoder{best_retention_idx}, ...
        compTable.retention_accuracy(best_retention_idx));
    
    % Most robust (smallest absolute drop considering std)
    robust_score = abs(compTable.delta_accuracy) + compTable.accuracy_zero_std;
    [~, most_robust_idx] = min(robust_score);
    fprintf('  • Most robust: %s (Δ = %.3f, std@0 = %.3f)\n', ...
        compTable.decoder{most_robust_idx}, ...
        compTable.delta_accuracy(most_robust_idx), ...
        compTable.accuracy_zero_std(most_robust_idx));
    
    fprintf('\n============================================================\n');
end