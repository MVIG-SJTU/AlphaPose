-------------------------------------------------------------------------------
-- Helpful functions for evaluation
-------------------------------------------------------------------------------

function loadPreds(predFile, doHm, doInp)
    local f = hdf5.open(projectDir .. '/exp/' .. predFile .. '.h5','r')
    local inp,hms
    local idxs = f:read('idxs'):all()
    local preds = f:read('preds'):all()
    if doHm then hms = f:read('heatmaps'):all() end
    if doInp then inp = f:read('input'):all() end
    return idxs, preds, hms, inp
end

function calcDists(preds, label, normalize)
    local dists = torch.Tensor(preds:size(2), preds:size(1))
    local diff = torch.Tensor(2)
    for i = 1,preds:size(1) do
        for j = 1,preds:size(2) do
            if label[i][j][1] > 1 and label[i][j][2] > 1 then
                dists[j][i] = torch.dist(label[i][j],preds[i][j])/normalize[i]
            else
                dists[j][i] = -1
            end
        end
    end
    return dists
end

function getPreds(hm)
    assert(hm:size():size() == 4, 'Input must be 4-D tensor')
    local max, idx = torch.max(hm:view(hm:size(1), hm:size(2), hm:size(3) * hm:size(4)), 3)
    local preds = torch.repeatTensor(idx, 1, 1, 2):float()
    preds[{{}, {}, 1}]:apply(function(x) return (x - 1) % hm:size(4) + 1 end)
    preds[{{}, {}, 2}]:add(-1):div(hm:size(4)):floor():add(1)
    return preds
end

function distAccuracy(dists, thr)
    -- Return percentage below threshold while ignoring values with a -1
    if not thr then thr = .5 end
    if torch.ne(dists,-1):sum() > 0 then
        return dists:le(thr):eq(dists:ne(-1)):sum() / dists:ne(-1):sum()
    else
        return -1
    end
end

function heatmapAccuracy(output, label, thr, idxs)
    -- Calculate accuracy according to PCK, but uses ground truth heatmap rather than x,y locations
    -- First value to be returned is average accuracy across 'idxs', followed by individual accuracies
    local preds = getPreds(output)
    local gt = getPreds(label)
    local dists = calcDists(preds, gt, torch.ones(preds:size(1))*opt.outputResH/10)
    local acc = {}
    local avgAcc = 0.0
    local badIdxCount = 0

    if not idxs then
        for i = 1,dists:size(1) do
            acc[i+1] = distAccuracy(dists[i])
    	    if acc[i+1] >= 0 then avgAcc = avgAcc + acc[i+1]
            else badIdxCount = badIdxCount + 1 end
        end
        if dists:size(1) ~= badIdxCount then
            acc[1] = avgAcc / (dists:size(1) - badIdxCount)
        else
            acc[1] = 0
        end
    else
        for i = 1,#idxs do
            acc[i+1] = distAccuracy(dists[idxs[i]])
	    if acc[i+1] >= 0 then avgAcc = avgAcc + acc[i+1]
            else badIdxCount = badIdxCount + 1 end
        end
        if #idxs ~= badIdxCount then
            acc[1] = avgAcc / (#idxs - badIdxCount)
        else
            acc[1] = 0
        end
    end
    return unpack(acc)
end

function basicAccuracy(output, label, thr)
    -- Calculate basic accuracy
    if not thr then thr = .5 end -- Default threshold of .5
    output = output:view(output:numel())
    label = label:view(label:numel())

    local rounded_output = torch.ceil(output - thr):typeAs(label)
    local eql = torch.eq(label,rounded_output):typeAs(label)

    return eql:sum()/output:numel()
end

function displayPCK(dists, part_idx, label, title, show_key)
    -- Generate standard PCK plot
    if not (type(part_idx) == 'table') then
        part_idx = {part_idx}
    end

    curve_res = 11
    num_curves = #dists
    local t = torch.linspace(0,.5,curve_res)
    local pdj_scores = torch.zeros(num_curves, curve_res)
    local plot_args = {}
    print(title)
    for curve = 1,num_curves do
        for i = 1,curve_res do
            t[i] = (i-1)*.05
            local acc = 0.0
            for j = 1,#part_idx do
                acc = acc + distAccuracy(dists[curve][part_idx[j]], t[i])
            end
            pdj_scores[curve][i] = acc / #part_idx
        end
        plot_args[curve] = {label[curve],t,pdj_scores[curve],'-'}
        print(label[curve],pdj_scores[curve][curve_res])
    end

    require 'gnuplot'
    gnuplot.raw('set title "' .. title .. '"')
    if not show_key then gnuplot.raw('unset key')
    else gnuplot.raw('set key font ",6" right bottom') end
    gnuplot.raw('set xrange [0:.5]')
    gnuplot.raw('set yrange [0:1]')
    gnuplot.plot(unpack(plot_args))
end
