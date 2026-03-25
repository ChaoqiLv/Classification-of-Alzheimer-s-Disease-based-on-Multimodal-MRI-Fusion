function [weights, lrate, sphere, data, signs, bias] = EpmljICA(data, ncomps, weights, chans, frames, lrate, maxsteps)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Initialize parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    degconst = 180 / pi;                % Conversion constant from radians to degrees
    datalength = frames;                % Total number of data frames
    DEFAULT_BLOWUP_in = 1e9;            % Threshold for weight blow-up
    DEFAULT_BLOWUP_FAC_in = 0.9;        % Learning-rate scaling factor when blow-up happens
    DEFAULT_RESTART_FAC_in = 0.9;       % Learning-rate scaling factor on restart
    annealdeg_in = 60;                  % Initial annealing angle (degrees)
    annealstep_in = 0.97;               % Learning-rate annealing step factor
    MAX_WEIGHT = 1e8;                   % Maximum allowed weight magnitude
    MIN_LRATE = 1e-6;                   % Minimum learning rate
    block = floor(sqrt(frames / 3));    % Block size (heuristic)
    nochange = 1e-6;                    % Stop training if weight change is below this threshold
    momentum = 0.9;                     % Momentum parameter
    biasflag = 1;                       % Whether to use bias
    extended = 0;                       % Whether to use extended ICA
    extblocks = 1;                      % Number of extension blocks
    nsub = 1;                           % Number of sub-samples
    wts_blowup = 0;                     % Flag indicating whether weights have blown up

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Compute sphering matrix %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    sphere = 2 * sqrtm(cov(data')) \ eye(size(data, 1));    % Sphering matrix
    data = sphere * data;                                   % Decorrelate electrode signals

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Initialize variables for weight updates %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    lastt = fix((datalength / block - 1) * block + 1);      % Last training index
    BI = eye(ncomps);                                       % Identity matrix

    startweights = weights;                                 % Store initial weights
    prevweights = startweights;                             % Weights from previous update
    oldweights = startweights;                              % Weights from previous step
    prevwtchange = zeros(ncomps, chans);                    % Previous weight change (for momentum)

    lrates = zeros(1, maxsteps);                            % Learning-rate history

    onesrow = ones(1, block);                               % Row vector of ones (used for bias)
    bias = zeros(ncomps, 1);                                % Bias vector
    signs = ones(1, ncomps);                                % Signs for extended ICA (if enabled)

    %%%%%%%%%%% Initialize sign matrix: first nsub elements set to -1, rest remain +1 %%%%%%%%%%%
    for k = 1:nsub
       signs(k) = -1;
    end
    signs = diag(signs);                                    % Convert signs vector to diagonal matrix

    urextblocks = extblocks;                                % Save original extblocks (for resets)

    step = 0;                                               % Global step counter
    blockno = 1;                                            % Block counter (currently unused)

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Main loop: Infomax ICA optimization %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    while step < maxsteps

        permute = randperm(datalength);                     % Shuffle data order each outer step

        %%%%%%%%%%% Process each training block %%%%%%%%%%%
        for t = 1:block:lastt

            if biasflag
                % Compute activation: y = WZ + b
                y = weights * data(:, permute(t:t+block-1)) + bias * onesrow;
            else
                y = weights * data(:, permute(t:t+block-1));
            end

            if ~extended
                % Apply logistic sigmoid nonlinearity
                sigmoid = 1 ./ (1 + exp(-y));

                % Compute phi(y) = 1 - 2y
                phi = 1 - 2 * sigmoid;

                % Update weight matrix W
                deltaW = lrate * (BI + (phi * y') / block) * weights;
                weights = weights + deltaW;

            else
                %%%%%%%%%%%%%%%%%%% Extended-ICA weight update %%%%%%%%%%%%%%%%%%%
                sigmoid = tanh(y);
                weights = weights + lrate * (BI - signs * sigmoid * y' - y * y') * weights;
            end

            %%%%%%%%%%%%%%%%%%%%%% Update bias vector b %%%%%%%%%%%%%%%%%%%%%%
            if biasflag

                if ~extended
                    deltaBias = lrate * mean(1 - 2 * sigmoid, 2);
                    bias = bias + deltaBias;
                else
                    %%%%%%%%%%%%%%%%%%% Extended-ICA bias update %%%%%%%%%%%%%%%%%%%
                    bias = bias + lrate * mean(-2 * sigmoid);
                end

            end

            %%%%%%%%%%%%%%%%%%%%%% Apply momentum update %%%%%%%%%%%%%%%%%%%%%%
            if momentum > 0
                weights = weights + momentum * prevwtchange;
                prevwtchange = weights - prevweights;
                prevweights = weights;
            end

            % Check whether weights have blown up
            if max(max(abs(weights))) > MAX_WEIGHT
                wts_blowup = 1;
                change = nochange;
            end

            blockno = blockno + 1;

            if wts_blowup
                break;
            end
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Compute weight changes and adjust learning rate %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if ~wts_blowup

           oldwtchange = weights - oldweights;
           step = step + 1;

           lrates(1, step) = lrate;

           delta = reshape(oldwtchange, 1, chans * ncomps);
           change = delta * delta';

        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Restart if weights blow up %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if wts_blowup || isnan(change) || isinf(change)

            fprintf('... Weights blow-up!!!\n');

            step = 0;
            wts_blowup = 0;
            blockno = 1;

            lrate = lrate * DEFAULT_RESTART_FAC_in;
            weights = startweights;
            oldweights = startweights;

            change = nochange;
            oldwtchange = zeros(ncomps, chans);
            delta = zeros(1, chans * ncomps);
            olddelta = delta;

            extblocks = urextblocks;

            prevweights = startweights;
            prevwtchange = zeros(ncomps, chans);

            lrates = zeros(1, maxsteps);
            bias = zeros(ncomps, 1);

            % Check whether learning-rate is still above the minimum
            if lrate > MIN_LRATE
                r = rank(data);
                if r < ncomps
                    fprintf('Data rank is %d. Cannot compute %d components.\n', r, ncomps);
                    return
                else
                    fprintf('Lowering learning rate to %g and restarting.\n', lrate);
                end
            else
                fprintf('Quitting - weight matrix may not be invertible!\n');
                return;
            end

        else

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Compute angle change and adjust learning rate %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            if step > 2

                angledelta = acos((delta * olddelta') / sqrt(change * oldchange));

                if degconst * angledelta > annealdeg_in
                    if lrate < 1e-5
                        annealstep_in = 0.98;  % Adjust annealing step factor
                    end
                    lrate = lrate * annealstep_in;  % Anneal learning rate
                    olddelta = delta;               % Accumulate angle changes
                    oldchange = change;           % Accumulate change magnitude
                end

            elseif step == 1
                olddelta = delta;
                oldchange = change;
            end

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Apply stopping rules %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            if step > 2 && change < nochange
                step = maxsteps;

            elseif change > DEFAULT_BLOWUP_in
                lrate = lrate * DEFAULT_BLOWUP_FAC_in;

            end

            % Update old weights
            oldweights = weights;

        end
    end
end