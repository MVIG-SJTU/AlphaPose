-- Prepare tensors for saving network output
local validSamples = opt.validIters * opt.validBatch
saved = {idxs = torch.Tensor(validSamples),
         preds = torch.Tensor(validSamples, unpack(ref.predDim))}
if opt.saveInput then saved.input = torch.Tensor(validSamples, unpack(ref.inputDim)) end
if opt.saveHeatmaps then saved.heatmaps = torch.Tensor(validSamples, unpack(ref.outputDim[1])) end

-- Main processing step
function step(tag)
    local avgLoss, avgAcc = 0.0, 0.0
    local output, err, idx
    local param, gradparam = model:getParameters()
    local function evalFn(x) return criterion.output, gradparam end

    if tag == 'train' then
        model:training()
        set = 'train'
    else
        model:evaluate()
        if tag == 'predict' then
            print("==> Generating predictions...")
            local nSamples = dataset:size('test')
            saved = {idxs = torch.Tensor(nSamples),
                     preds = torch.Tensor(nSamples, unpack(ref.predDim))}
            if opt.saveInput then saved.input = torch.Tensor(nSamples, unpack(ref.inputDim)) end
            if opt.saveHeatmaps then saved.heatmaps = torch.Tensor(nSamples, unpack(ref.outputDim[1])) end
            set = 'test'
        else
            set = 'valid'
        end
    end

    local nIters = opt[set .. 'Iters']
    local actualIters = nIters
    local iterSize = opt.iterSize

    local dataloader = loader[set]:run()
    for i,sample in dataloader do
        xlua.progress(i, nIters)

        if tag == 'train' then
            -- Training: Do backpropagation and optimization
            local loss_val = 0.0
            local batch_acc = 0.0
            model:zeroGradParameters()
            if i+iterSize > nIters then iterSize = nIters-i+1 end
            for j = 1,iterSize do
                if j ~= 1 then
                    i,sample = dataloader()
                    xlua.progress(i, nIters)
                end
                local input, label, mask, _ = unpack(sample)
                if opt.GPU ~= -1 then
                    -- Convert to CUDA
                    input = applyFn(function (x) return x:cuda() end, input)
                    label = applyFn(function (x) return x:cuda() end, label)
                    mask = applyFn(function (x) return x:cuda() end, mask)
                end
                local output = model:forward(input)
                output = applyFn(function (x) return x:clone() end, output)
                for k=1,opt.nStack do output[k]:cmul(mask) end
                
                --- hard mining
                if opt.hardMining == true then
                    local part_output, part_label, part_loss
                    local loss = torch.zeros(17)
                    local partloss = {}
                    for part=1, dataset.nJoints_coco do
                        part_output = applyFn(function (x) return x:narrow(2,part,1):clone() end, output)
                        part_label = applyFn(function (x) return x:narrow(2,part,1):clone() end, label)
                        loss[part] = criterion:forward(part_output,part_label)
                    end
                    _, lowlossIndex = torch.topk(loss, 9, 1) -- lowest 9 loss (half of the total part)
                    for tmpI=1,9 do
                        local part = lowlossIndex[tmpI]
                        partloss[part] = 1
                        output = applyFn(function (x,y) x:narrow(2, part, 1):copy(y:narrow(2, part,1)) return x:clone() end, output, label)
                    end
                    --partLog:add(partloss)
                end

                tmp_loss = criterion:forward(output, label)
                loss_val = loss_val + tmp_loss
                batch_acc = batch_acc + accuracy(output, label)
                model:backward(input, criterion:backward(output, label))
            end
            if iterSize ~= 0 then
                loss_val = loss_val / iterSize
                gradparam:mul(1.0 / iterSize)
                optfn(function() return loss_val, gradparam end, param, optimState)
                avgLoss = avgLoss + loss_val*iterSize / actualIters
                -- Calculate accuracy
                avgAcc = avgAcc + batch_acc / actualIters
            end
        else
            local input, label, mask, indices = unpack(sample)
            if opt.GPU ~= -1 then
                -- Convert to CUDA
                input = applyFn(function (x) return x:cuda() end, input)
                label = applyFn(function (x) return x:cuda() end, label)
                mask = applyFn(function (x) return x:cuda() end, mask)
            end
            local output = model:forward(input)
            output = applyFn(function (x) return x:clone() end, output)
            for k=1,opt.nStack do output[k]:cmul(mask) end

            local err = criterion:forward(output, label)
            avgLoss = avgLoss + err / nIters

            -- Validation: Get flipped output
            local flippedOut = model:forward(flip(input))
            flippedOut = applyFn(function (x) return flip(shuffleLR(x)) end, flippedOut)
            output = applyFn(function (x,y) return x:add(y):div(2) end, output, flippedOut)
            for k=1,opt.nStack do output[k]:cmul(mask) end
            -- Calculate accuracy
            --avgAcc = avgAcc + accuracyCOCO(output, label) / nIters
            avgAcc = avgAcc + accuracy(output, label) / nIters
            -- Save sample
            local bs = opt[set .. 'Batch']
            local tmpIdx = (i-1) * bs + 1
            local tmpOut = output
            if type(tmpOut) == 'table' then tmpOut = output[#output] end
            tmpOut = tmpOut:narrow(2,1,17):clone()
            if opt.saveInput then saved.input:sub(tmpIdx, tmpIdx+bs-1):copy(input) end
            if opt.saveHeatmaps then saved.heatmaps:sub(tmpIdx, tmpIdx+bs-1):copy(tmpOut) end
            saved.idxs:sub(tmpIdx, tmpIdx+bs-1):copy(indices)
            saved.preds:sub(tmpIdx, tmpIdx+bs-1):copy(postprocess(set,indices,output))
        end
    end


    -- Print and log some useful metrics
    print(string.format("      %s : Loss: %.7f Acc: %.4f", set, avgLoss, avgAcc))
    if ref.log[set] then
        table.insert(opt.acc[set], avgAcc)
        ref.log[set]:add{
            ['epoch     '] = string.format("%d" , epoch),
            ['loss      '] = string.format("%.6f" , avgLoss),
            ['acc       '] = string.format("%.4f" , avgAcc),
            ['LR        '] = string.format("%g" , optimState.learningRate)
        }
    end

    if (tag == 'valid' and opt.snapshot ~= 0 and epoch % opt.snapshot == 0) or tag == 'predict' then
        -- Take a snapshot
        model:clearState()
        -- Update lastEpoch info before we save
        opt.lastEpoch = epoch
        torch.save(paths.concat(opt.save, 'options.t7'), opt)
        torch.save(paths.concat(opt.save, 'optimState.t7'), optimState)
        if torch.type(model) == 'nn.DataParallelTable' then
            torch.save(paths.concat(opt.save,'model_' .. epoch .. '.t7'), cleanDPT(model))
        else
            torch.save(paths.concat(opt.save,'model_' .. epoch .. '.t7'), model)
        end
        local predFilename = 'preds.h5'
        if tag == 'predict' then predFilename = 'final_' .. predFilename end
        local predFile = hdf5.open(paths.concat(opt.save,predFilename),'w')
        for k,v in pairs(saved) do predFile:write(k,v) end
        predFile:close()
    end
end

function train() step('train') end
function valid() step('valid') end
function predict() step('predict') end
