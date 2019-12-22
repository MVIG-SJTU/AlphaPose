local threads = require 'threads'
local argv = arg
require 'paths'
require 'nn'
nnlib = nn
paths.dofile('Logger.lua')
paths.dofile('util.lua')
paths.dofile('img.lua')


cutorch.setDevice(1)
local a,preds,scores,nsamples,idxs,prog

if #argv ~= 6 then
    print("Usage: th main-alpha-pose.lua ${MODE} ${INPUT_PATH} ${OUTPUT_PATH} ${GPU_NUM} ${BATCH_SIZE} ${DATASET}, current argvs number is "..tostring(#argv))
    print("    MODE: demo or predict, demo: display result for each person, predict: generate predicitions")
    print("    INPUT_PATH: the folder of the input images")
    print("    OUTPUT_PATH - the bbox is stored at OUTPUT_PATH/BBOX/ and the pose will be stored to OUTPUT_PATH/POSE/")
    print("    GPU_NUM: the number of the gpus")
    print("    BATCH_SIZE: batch size")
    print("    DATASET: dataset format to use, 'COCO' or 'MPII'")
    return
end

local out_format = argv[6] -- 'COCO' or 'MPII'
print(out_format)
a = loadAnnotations(argv[3]..'/BBOX')
idxs = torch.range(1,a.nsamples)
nsamples = idxs:nElement() 
local shuffleLR
if out_format == 'COCO' then
    preds = torch.Tensor(nsamples,17,2)
    scores = torch.Tensor(nsamples,17,1)
    shuffleLR = shuffleLRCOCO
elseif out_format == 'MPII' then
    preds = torch.Tensor(nsamples,16,2)
    scores = torch.Tensor(nsamples,16,1)
    shuffleLR = shuffleLRMPII
else
    print("out_format must be either 'COCO' or 'MPII'")
    return
end

function init(idx)
    require 'paths'
    require 'nn'
    nnlib = nn
    paths.dofile('util.lua')
    paths.dofile('img.lua')

    cutorch.setDevice(1)
end


local divNum = tonumber(argv[4])
local batchSize = tonumber(argv[5])
prog = torch.zeros(divNum)
--------------------------------------------------------------------------------
-- Main loop
--------------------------------------------------------------------------------
local function loop(startIndex,endIndex,k)
    --------------------------------------------------------------------------------
    -- Initialization
    --------------------------------------------------------------------------------
    function applyFn(fn, t, t2)
        -- Apply an operation whether passed a table or tensor
        local t_ = {}
        if type(t) == "table" then
            if t2 then
                for i = 1,#t do t_[i] = applyFn(fn, t[i], t2[i]) end
            else
                for i = 1,#t do t_[i] = applyFn(fn, t[i]) end
            end
        else t_ = fn(t, t2) end
        return t_
    end

    function printProgress(k,cur,num)
        s1 = ''
        for j=1,k-1 do
            s1 = s1 .. '\n'
        end
        s1 = s1 .. '\r\b'
        s2 = ''
        for j=1,k do
            s2 = s2 .. "\r\b"
        end
        print(s1)
        xlua.progress(cur,num)
        print("\r " .. k .. ":"..s2)
    end


    local m = torch.load('models/final_model.t7')  -- 6 is 71.9mAP 8:71.8

    for i = startIndex,endIndex,batchSize do
        -- Set up input image
        if string.sub(argv[2],-1) ~= '/' then
                argv[2] = argv[2]..'/'
        end
        local minibat = batchSize - math.max(0, i+batchSize-endIndex-1)
        local inputResH = 320
        local inputResW = 256
        local outResH = 80
        local outResW = 64
        
        local inp = torch.zeros(minibat, 3, inputResH, inputResW)
        local pt1 = torch.zeros(minibat, 2)
        local pt2 = torch.zeros(minibat, 2)
        for batId = 1,minibat do
            local tmpInp, tmpPt1, tmpPt2 = loadInp(argv, a, idxs, i+batId-1)
            inp[batId]:copy(tmpInp)
            pt1[batId]:copy(tmpPt1)
            pt2[batId]:copy(tmpPt2)
        end
        -- Get network output
        local out = m:forward(inp:cuda())
        out = applyFn(function (x) return x:clone() end, out)
        for i =1,8 do
            if out_format == 'COCO' then
                out[i] = out[i]:narrow(2,1,17):clone()
            elseif out_format == 'MPII' then
                out[i] = out[i]:narrow(2,18,16):clone()
            end
        end
        --flip
        local flippedOut = m:forward(flip(inp:cuda()))
        for i=1,8 do
            if out_format == 'COCO' then
                flippedOut[i] = flippedOut[i]:narrow(2,1,17):clone()
            elseif out_format == 'MPII' then
                flippedOut[i] = flippedOut[i]:narrow(2,18,16):clone()
            end
        end
        flippedOut = applyFn(function (x) return flip(shuffleLR(x)) end, flippedOut)
        out = applyFn(function (x,y) return x:add(y):div(2) end, out, flippedOut)
        
        cutorch.synchronize()
        --local preds_hm, preds_img, pred_scores
        hm = out[8]:float()
        hm[hm:lt(0)] = 0
        if argv[6] == 'COCO' then
             local g = image.gaussian(4*1 + 1)
             for bId = 1,minibat do
               local s = image.convolve(hm[bId],g,'same')
               hm[bId] = s:clone()
             end
        end
        -- Get predictions (hm and img refer to the coordinate space)
        local preds_hm, preds_img, pred_scores = getPreds(hm, pt1:int(), pt2:int(),inputResH,inputResW,outResH,outResW)

        preds:narrow(1,i,minibat):copy(preds_img)
        scores:narrow(1,i,minibat):copy(pred_scores)
        
        prog[k] = i-startIndex+1
        --printProgress(k,i-startIndex+1,endIndex-startIndex+1)
        xlua.progress(i-startIndex+1,endIndex-startIndex+1)
        -- Display the result
        if argv[1] == 'demo' then
            preds_hm:mul(inputResH/outResH) -- Change to input scale
            local dispImg = drawOutputcoco(inp, hm, preds_hm[1],inputResH,inputResW,outResH,outResW)
            w = image.display{image=dispImg,win=w}
            sys.sleep(3)
        end
        
        collectgarbage()
    end
    return prog,preds,scores
end

local pools = threads.Threads(divNum,init)
local gpuNum = cutorch.getDeviceCount()

if divNum == 1 then
    cutorch.setDevice(1)
    loop(1,nsamples,1)
else
    for i = 1,divNum do
        pools:addjob(
            function (i) 
                cutorch.setDevice(math.ceil(i/math.ceil(divNum/gpuNum)))
                startIndex = (i-1)*(math.ceil(nsamples/divNum))+1
                endIndex = math.min(nsamples, i*(math.ceil(nsamples/divNum)))
                tP,tPreds,tScores = loop(startIndex,endIndex,i)
                return tP,tPreds,tScores,i,startIndex,endIndex
            end,
            function (tP,tPreds,tScores,i,startIndex,endIndex)
                prog[i] = tP[i]
                for k=startIndex,endIndex do
                    preds[k] = tPreds[k]
                    scores[k] = tScores[k]
                end
            end,
            i)
    end
end

pools:synchronize()
pools:terminate()
print("----------Finished----------")
-- Save predictions
if argv[1] == 'predict' then
    local predFile = hdf5.open(argv[3]..'/POSE/test-pose.h5', 'w')
    predFile:write('preds', preds)
    predFile:write('scores',scores)
    predFile:close()
elseif argv[1] == 'demo' then
    w.window:close()
end
