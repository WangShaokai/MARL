function [ pred, accs, logLs ] = MARL(fea, link, gnd, trainIdx, testIdx, alpha, numK)
    %% parameters
    num_label = size(gnd, 2);
%     Py_d = NBC_mvplsa3(gnd', fea, link, testIdx, trainIdx, alpha, numK);
    [Py_d, logLs, Pz_ds, accs] = MARL_GMLGCC(gnd', fea, link, testIdx, trainIdx, alpha, numK);
    %% assign labels
    [~, pred_label] = max(Py_d, [], 1);
    pred = zeros(sum(testIdx), num_label);
    ind = sub2ind(size(pred), 1:sum(testIdx), pred_label(testIdx));
    pred(ind) = 1;
end

