function [weights, A, W, sources] = optimizeinfomax(datamatrix, ncomps, max_steps, num_mods, nochange, lrate, ann_stp, verbose)

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Split dataset %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Split according to data dimensions
    data_sMRI_GM = datamatrix(:, 1:44347);                           % 664 * 44347
    data_DTI_FA = datamatrix(:, 44348:28774+44347);                  % 664 * 28774
    data_DTI_MD = datamatrix(:, 44347+28774+1:44347+2*28774);        % 664 * 28774
    data_fMRI_IC = datamatrix(:, 44347+2*28774+1:end);               % 664 * 44347

    % Print data dimensions for each modality
    % disp('Data dimensions after splitting:');
    % disp(['sMRI_GM: ', num2str(size(data_sMRI_GM))]);
    % disp(['DTI_FA: ', num2str(size(data_DTI_FA))]);
    % disp(['DTI_MD: ', num2str(size(data_DTI_MD))]);
    % disp(['fMRI_IC: ', num2str(size(data_fMRI_IC))]);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PCA dimensionality reduction %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    disp('Perform PCA dimensionality reduction for each modality...');
    [whitesigGM, dewhiteMGM] = icatb_calculate_pca(data_sMRI_GM', ncomps);
    [whitesigFA, dewhiteMFA] = icatb_calculate_pca(data_DTI_FA', ncomps);
    [whitesigMD, dewhiteMMD] = icatb_calculate_pca(data_DTI_MD', ncomps);
    [whitesigIC, dewhiteMIC] = icatb_calculate_pca(data_fMRI_IC', ncomps);

    disp('Output whitening matrix sizes after PCA for each modality:');
    disp(['sMRI_GM: ', num2str(size(whitesigGM))]);      % 44347*18
    disp(['DTI_FA: ', num2str(size(whitesigFA))]);       % 28774*18
    disp(['DTI_MD: ', num2str(size(whitesigMD))]);       % 28774*18
    disp(['fMRI_IC: ', num2str(size(whitesigIC))]);      % 44347*18

    % disp('Output dewhitening matrix sizes after PCA:');
    % disp(['sMRI_GM: ', num2str(size(dewhiteMGM))]);
    % disp(['DTI_FA: ', num2str(size(dewhiteMFA))]);
    % disp(['DTI_MD: ', num2str(size(dewhiteMMD))]);
    % disp(['fMRI_IC: ', num2str(size(dewhiteMIC))]);

    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% (Optional) Reconstruct signal after PCA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % % Attempt to approximate the original signal
    % disp('Approximate reconstruction of the original signal');
    % sMRI_GM_restoring = whitesigGM * dewhiteMGM';
    % DTI_FA_restoring = whitesigFA * dewhiteMFA';
    % DTI_MD_restoring = whitesigMD * dewhiteMMD';
    % fMRI_IC_restoring = whitesigIC * dewhiteMIC';
    %
    % % Validate whether separated data matches the original data
    % disp('Validate whether reconstructed data matches pre-PCA data:');
    % disp(['GM identical: ', num2str(isequal(data_sMRI_GM', sMRI_GM_restoring))]);
    % disp(['FA identical: ', num2str(isequal(data_DTI_FA', DTI_FA_restoring))]);
    % disp(['MD identical: ', num2str(isequal(data_DTI_MD', DTI_MD_restoring))]);
    % disp(['ICs identical: ', num2str(isequal(data_fMRI_IC', fMRI_IC_restoring))]);
    %
    % % Check reconstruction error
    % disp('Reconstruction error before and after PCA:');
    % reconstruction_error_GM = norm(data_sMRI_GM' - sMRI_GM_restoring, 'fro');
    % fprintf('GM reconstruction error: %.4f\n', reconstruction_error_GM);
    %
    % reconstruction_error_FA = norm(data_DTI_FA' - DTI_FA_restoring, 'fro');
    % fprintf('FA reconstruction error: %.4f\n', reconstruction_error_FA);
    %
    % reconstruction_error_MD = norm(data_DTI_MD' - DTI_MD_restoring, 'fro');
    % fprintf('MD reconstruction error: %.4f\n', reconstruction_error_MD);
    %
    % reconstruction_error_IC = norm(data_fMRI_IC' - fMRI_IC_restoring, 'fro');
    % fprintf('IC reconstruction error: %.4f\n', reconstruction_error_IC);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Get channels and frames after PCA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % chans frames
    dataGM = whitesigGM';    % 18*44347
    dataFA = whitesigFA';    % 18*28774
    dataMD = whitesigMD';    % 18*28774
    dataIC = whitesigIC';    % 18*44347

    [chansGM, framesGM] = size(dataGM);
    [chansFA, framesFA] = size(dataFA);
    [chansMD, framesMD] = size(dataMD);
    [chansIC, framesIC] = size(dataIC);

    disp('Output number of channels and frames for each modality:');
    disp(['Get the number of channels and frames of GM : ', num2str(size(dataGM))]);
    disp(['Get the number of channels and frames of FA : ', num2str(size(dataFA))]);
    disp(['Get the number of channels and frames of MD : ', num2str(size(dataMD))]);
    disp(['Get the number of channels and frames of IC : ', num2str(size(dataIC))]);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Weight optimization %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Initialize weight matrices: weights = 18 * 18

    % Reset random number generator to default mode
    rng('default');
    rng(42);  % Set random seed

    weightsGM = randsmall(ncomps, chansGM);
    weightsFA = randsmall(ncomps, chansFA);
    weightsMD = randsmall(ncomps, chansMD);
    weightsIC = randsmall(ncomps, chansIC);

    %%%%%%%%%%% Variables for weight blow-up handling after initialization %%%%%%%%%%%
    DEFAULT_BLOWUP = 1e9;               % Weight blow-up threshold
    DEFAULT_BLOWUP_FAC = 0.9;           % Learning rate blow-up factor
    wts_passed = 0;                     % Weight passing flag
    annealdeg = 60;                     % Annealing angle
    degconst = 180 / pi;                % Radians to degrees conversion constant
    pass = 1;                           % Iteration counter
    alpha = 0.5;                        % Smoothing factor
    min_weight = 0.05;                  % Lower bound of weights
    prev_weights = zeros(num_mods, 1);  % Initialize history weights
    posactflag = 'on';
    annealstep = 0.97;

    %%%%%%%%%%%% Display debug figure %%%%%%%%%%%
    figure,

    %%%%%%%%%%%%%%%%%%%% Joint optimization %%%%%%%%%%%%%%%%%%%%
    while pass <= max_steps
        % Slow optimization of data matrices
        if pass == 1
            maxsteps = 1;
        elseif pass > 1 && pass < 21
            maxsteps = 4;
        elseif pass >= 21 && pass < 41
            maxsteps = 8;
        elseif pass >= 41 && pass < 61
            maxsteps = 16;
        elseif pass >= 61 && pass < 81
            maxsteps = 32;
        elseif pass >= 81
            maxsteps = 64;
        end

        %%%%%%%%%% Update weights for each modality separately %%%%%%%%%%
        [W1, lrate, ~] = EpmljICA(dataGM, ncomps, weightsGM, chansGM, framesGM, lrate, maxsteps);
        [W2, lrate, ~] = EpmljICA(dataFA, ncomps, weightsFA, chansFA, framesFA, lrate, maxsteps);
        [W3, lrate, ~] = EpmljICA(dataMD, ncomps, weightsMD, chansMD, framesMD, lrate, maxsteps);
        [W4, lrate, ~] = EpmljICA(dataIC, ncomps, weightsIC, chansIC, framesIC, lrate, maxsteps);

        % Calculate kurtosis for each modality
        sources1 = W1 * dataGM;
        kurtosis1 = mean(abs(kurtosis(sources1, [], 2) - 3));
        sources2 = W2 * dataFA;
        kurtosis2 = mean(abs(kurtosis(sources2, [], 2) - 3));
        sources3 = W3 * dataMD;
        kurtosis3 = mean(abs(kurtosis(sources3, [], 2) - 3));
        sources4 = W4 * dataIC;
        kurtosis4 = mean(abs(kurtosis(sources4, [], 2) - 3));

        % Normalize original weights
        total_kurt = kurtosis1 + kurtosis2 + kurtosis3 + kurtosis4;
        if total_kurt == 0
            total_kurt = 1;
        end
        w1 = kurtosis1 / total_kurt;
        w2 = kurtosis2 / total_kurt;
        w3 = kurtosis3 / total_kurt;
        w4 = kurtosis4 / total_kurt;

        % Apply EMA smoothing
        if pass > 1
            w_current = [w1; w2; w3; w4];
            w_smoothed = alpha * w_current + (1 - alpha) * prev_weights;
        else
            w_smoothed = [w1; w2; w3; w4];
        end

        % Apply lower bound to weights
        w_clipped = max(w_smoothed, min_weight);
        w_normalized = w_clipped / sum(w_clipped);

        % Update weights
        w1 = w_normalized(1);
        w2 = w_normalized(2);
        w3 = w_normalized(3);
        w4 = w_normalized(4);
        prev_weights = [w1; w2; w3; w4];

        % Weighted average
        weights = w1 * W1 + w2 * W2 + w3 * W3 + w4 * W4;

        % Update each modality's weights
        weightsGM = weights;
        weightsFA = weights;
        weightsMD = weights;
        weightsIC = weights;

        %%%%%%%%%% Adjust learning rate according to weight change %%%%%%%%%%
        % Calculate weight change
        if pass == 1
            % Initialize weights
            oldweights = weights;

            % Compute initial weight change
            oldwtchangeGM = oldweights - W1;
            olddeltaGM = reshape(oldwtchangeGM, 1, chansGM * ncomps);
            changeGM = olddeltaGM * olddeltaGM';

            oldwtchangeFA = oldweights - W2;
            olddeltaFA = reshape(oldwtchangeFA, 1, chansFA * ncomps);
            changeFA = olddeltaFA * olddeltaFA';

            oldwtchangeMD = oldweights - W3;
            olddeltaMD = reshape(oldwtchangeMD, 1, chansMD * ncomps);
            changeMD = olddeltaMD * olddeltaMD';

            oldwtchangeIC = oldweights - W4;
            olddeltaIC = reshape(oldwtchangeIC, 1, chansIC * ncomps);
            changeIC = olddeltaIC * olddeltaIC';

            % Compute average of initial weight changes
            oldchange = (changeGM + changeFA + changeMD + changeIC) / 4;
            olddelta = (olddeltaIC + olddeltaMD + olddeltaFA + olddeltaGM) / 4;
            change = 999;  % Large initial change value

        else

            % Compute weight change
            oldwtchange = weights - oldweights;
            delta = reshape(oldwtchange, 1, chansGM * ncomps);
            change = delta * delta';
            angledelta = acos((delta * olddelta') / sqrt(change * oldchange));

            % Adjust learning rate
            if degconst * angledelta > annealdeg
                if lrate < 0.00001
                    annealstep = ann_stp;
                end
                lrate = lrate * annealstep;
                olddelta = delta;
                oldchange = change;
            end

            fprintf('step %d - lrate %7.9f, wchange %7.7f, angledelta %4.1f deg\n', pass, lrate, change, degconst * angledelta);
            oldweights = weights;

            %%%%%%%%%% Display optimization process %%%%%%%%%%
            if pass == 1
                plt_change = zeros(1, max_steps);
            end

            if verbose == 1
                if pass == 2
                    plt_change(pass) = change;
                    plot(2:pass, plt_change(2:pass), '-o', 'Color', [0 0.4470 0.7410], 'LineWidth', 1.5, 'MarkerSize', 6);
                    set(gca, 'XTick', [1:50:max_steps, 500], 'FontSize', 12, 'FontName', 'Times New Roman');
                    xlim([0, max_steps]);
                    ylim([0, max(plt_change) * 1.1]);
                    grid on;
                    title('Weight Changes During Joint Optimization', 'FontSize', 14, 'FontWeight', 'bold', 'FontName', 'Times New Roman');
                    xlabel('Epoch', 'FontSize', 14, 'FontName', 'Times New Roman');
                    ylabel('Weight Change', 'FontSize', 14, 'FontName', 'Times New Roman');
                    hold on;

                elseif pass < max_steps
                    plt_change(pass) = change;
                    plot(2:pass, plt_change(2:pass), '-o', 'Color', [0 0.4470 0.7410], 'LineWidth', 1.5, 'MarkerSize', 6);
                    set(gca, 'XTick', [1:50:max_steps, 500], 'FontSize', 12, 'FontName', 'Times New Roman');
                    xlim([0, max_steps]);
                    ylim([0, max(plt_change) * 1.1]);
                    pause(0.01);
                    grid on;
                    title('Weight Changes During Joint Optimization', 'FontSize', 14, 'FontWeight', 'bold', 'FontName', 'Times New Roman');
                    xlabel('Epoch', 'FontSize', 12, 'FontName', 'Times New Roman');
                    ylabel('Weight Change', 'FontSize', 12, 'FontName', 'Times New Roman');

                elseif pass == max_steps
                    plt_change(pass) = change;
                    save('plt_change_final.mat', 'plt_change');

                    plot(2:pass, plt_change(2:pass), '-o', 'Color', [0 0.4470 0.7410], 'LineWidth', 1.5, 'MarkerSize', 6);
                    set(gca, 'XTick', [1:50:max_steps, 500], 'FontSize', 14, 'FontName', 'Times New Roman');
                    xlim([0, max_steps]);
                    ylim([0, max(plt_change) * 1.1]);
                    grid on;
                    title('Weight Changes During Joint Optimization', 'FontSize', 14, 'FontWeight', 'bold', 'FontName', 'Times New Roman');
                    xlabel('Epoch', 'FontSize', 14, 'FontName', 'Times New Roman');
                    ylabel('Weight Change', 'FontSize', 14, 'FontName', 'Times New Roman');

                    % Save final figure as high-resolution PNG
                    print('weights_optimization_final', '-dpng', '-r300');
                end
            end

        end

        pass = pass + 1;

        %%%%%%%%%% Check stopping condition %%%%%%%%%%
        if pass > 2 && change < nochange
            laststep = pass;
            disp(['Change is negligible; the iteration number is: ', num2str(laststep)]);
            pass = max_steps + 1;
        elseif change > DEFAULT_BLOWUP
            lrate = lrate * DEFAULT_BLOWUP_FAC;
        end
    end

    if verbose == 1
        hold off;
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Reconstruct signal and compute mixing matrix %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [whitesigGmCm, dewhiteMGmCm, ~, ~, ~] = icatb_calculate_pca(datamatrix', ncomps);
    dataGmCm = whitesigGmCm';

    %%%%%%%%%%%%%%%%%%%% Compute sphering matrix %%%%%%%%%%%%%%%%%%%%
    sphereGmCm = 2 * sqrtm(cov(dataGmCm')) \ eye(size(dataGmCm, 1));
    data = sphereGmCm * dataGmCm;

    %%%%%%%%%%%%%%%%%%%% Ensure components are oriented to positive activations %%%%%%%%%%%%%%%%%%%%
    if strcmp(posactflag, 'on')
        % Reorient the activation values to all-positive RMS and accordingly adjust weights
        [activations, ~, weights] = icatb_posact(data, weights);
    else
        activations = weights * data;
    end

    disp('Activations (size):');
    disp(size(activations));

    %%%%%%%%%%%%%%%%%%%% Sort components %%%%%%%%%%%%%%%%%%%%
    % Sort components in descending order by projection variance
    fprintf('Rank order components by descending projection variance...\n');
    if wts_passed == 0
        meanvar = zeros(ncomps, 1);
        if ncomps == chansGM
            winv = inv(weights * sphereGmCm);
        else
            fprintf('Using pseudo-inverse of weight matrix to rank order component projections.\n');
            winv = pinv(weights * sphereGmCm);
        end

        for s = 1:ncomps
            fprintf('%d ', s);
            compproj = winv(:, s) * activations(s, :);
            meanvar(s) = mean(sum(compproj .* compproj) / (size(compproj, 1) - 1));
        end

        fprintf('\n');

        %%%%%%%%%%%%%%%%%%%% Sort components by mean variance %%%%%%%%%%%%%%%%%%%%
        [~, windex] = sort(meanvar);
        windex = windex(ncomps:-1:1);
        meanvar = meanvar(windex);

        % Reorder activations and weights
        fprintf('\nReorder activation waveforms...\n');
        activations = activations(windex, :);

        disp('Sorted activations (size):');
        disp(size(activations));

        weights = weights(windex, :);

    else
        fprintf('Components are not sorted by variance.\n');
    end

    %%%%%%%%%%%%%%%%%%%% Compute mixing matrix and source signals %%%%%%%%%%%%%%%%%%%%
    % Introduce data transforms directly into weights to adjust weights to the whitened data
    sphere_weights = weights * sphereGmCm;
    disp('Size of mixing matrix weights:');
    disp(size(sphere_weights));
    save('weights.mat', 'sphere_weights');

    A = dewhiteMGmCm * pinv(sphere_weights);
    disp('Size of mixing matrix A:');
    disp(size(A));
    save('A.mat', 'A');

    W = pinv(A);
    disp('Size of unmixing matrix W:');
    disp(size(W));

    sources = sphere_weights * dataGmCm;
    disp('Size of source signals:');
    disp(size(sources));
    save('sources.mat', 'sources');

    X = A * sources;

    disp('Reconstructed observation signals (X):');
    reconstruction_error_X = norm(datamatrix - X, 'fro');
    fprintf('Reconstruction error of X: %.4f\n', reconstruction_error_X);

end