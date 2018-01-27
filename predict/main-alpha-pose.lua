require 'paths'

require 'nn'
nnlib = nn
paths.dofile('util.lua')
paths.dofile('img.lua')

cutorch.setDevice(1)

--------------------------------------------------------------------------------
-- Initialization
--------------------------------------------------------------------------------

if arg.size[0] != 3 then
    print("Usage: th main-alpha-pose.lua ${MODE} ${INPUT_PATH} ${OUTPUT_PATH}")
    print("    MODE: demo or predict, demo: display result for each person, predict: generate predicitions")
    print("    INPUT_PATH: the folder of the input images")
    print("    OUTPUT_PATH - the bbox is stored at OUTPUT_PATH/BBOX/ and the pose will be stored to OUTPUT_PATH/POSE/")
    return
end

a = loadAnnotations(arg[3]..'/BBOX')

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

m = torch.load('model_146.t7')   -- 115 is 71.2mAP

if arg[1] == 'demo' then
    --idxs = torch.Tensor({695, 3611, 2486, 7424, 10032, 5, 4829})
    -- If all the MPII images are available, use the following line to see a random sampling of images
    --idxs = torch.randperm(a.nsamples):sub(1,50)
    idxs = torch.range(1,a.nsamples)
else
    idxs = torch.range(1,a.nsamples)
end

if arg[1] == 'eval' then
    nsamples = 0
else
    nsamples = idxs:nElement() 
    -- Displays a convenient progress bar
    xlua.progress(0,nsamples)
    preds = torch.Tensor(nsamples,17,2)
    scores = torch.Tensor(nsamples,17,1)
end

--------------------------------------------------------------------------------
-- Main loop
--------------------------------------------------------------------------------

for i = 1,nsamples do
    -- Set up input image

    --local im = image.load('data/OutdoorPoseDataset/' .. a['images'][idxs[i]])
    local im
    if string.sub(arg[2],-1) ~= '/' then
            arg[2] = arg[2]..'/'
    end
    if arg[1] == 'valid' or arg[1] == 'demo' then
       im = image.load(arg[2] .. a['images'][idxs[i]],3)
    else
       im = image.load(arg[2] .. a['images'][idxs[i]],3)
    end
    im[1]:add(-0.406)
    im[2]:add(-0.457)
    im[3]:add(-0.480)
    local imght = im:size()[2]
    local imgwidth = im:size()[3]
    local pt1= torch.Tensor(2)
    local pt2= torch.Tensor(2)
    pt1[1] = a['xmin'][idxs[i]]
    pt1[2] = a['ymin'][idxs[i]]
    pt2[1] = a['xmax'][idxs[i]]
    pt2[2] = a['ymax'][idxs[i]]
    local ht = pt2[2]-pt1[2]
    local width = pt2[1]-pt1[1]

    local scaleRate
    if width > 100 then
       scaleRate = 0.2 
    else
      scaleRate = 0.3 
    end

    local bias=0
    local rand = torch.rand(1)
    pt1[1] = math.max(0,(pt1[1] - width*scaleRate/2 - rand*width*bias)[1])
    pt1[2] = math.max(0,(pt1[2] - ht*scaleRate/2 - rand*ht*bias)[1])
    pt2[1] = math.max(math.min(imgwidth,(pt2[1] + width*scaleRate/2 + (1-rand)*width*bias)[1]),pt1[1]+5)
    pt2[2] = math.max(math.min(imght,(pt2[2] + ht*scaleRate/2 + (1-rand)*ht*bias)[1]),pt1[2]+5)
    local inputRes = 256

    --local inp = crop(im, center, scale, 0, inputRes)
    local inp = cropBox(im, pt1:int(), pt2:int(), 0, inputRes)
    -- Get network output
    local out = m:forward(inp:view(1,3,inputRes,inputRes):cuda())
    --flip
    out = applyFn(function (x) return x:clone() end, out)
    for i =1,8 do
      out[i] = out[i]:narrow(2,1,17):clone()
    end
    local flippedOut = m:forward(flip(inp:view(1,3,inputRes,inputRes):cuda()))
    for i=1,8 do
      flippedOut[i] = flippedOut[i]:narrow(2,1,17):clone()
    end
    flippedOut = applyFn(function (x) return flip(shuffleLRCOCO(x)) end, flippedOut)
    out = applyFn(function (x,y) return x:add(y):div(2) end, out, flippedOut)
    

    cutorch.synchronize()

    local hm = out[8][1]:float()

    hm[hm:lt(0)] = 0


    -- Get predictions (hm and img refer to the coordinate space)
    local preds_hm, preds_img, pred_scores = getPreds(hm, pt1:int(), pt2:int())

    preds[i]:copy(preds_img)
    scores[i]:copy(pred_scores)

    xlua.progress(i,nsamples)

    -- Display the result
    if arg[1] == 'demo' then
        preds_hm:mul(inputRes/64) -- Change to input scale
        local dispImg = drawOutputcoco(inp, hm, preds_hm[1])
        w = image.display{image=dispImg,win=w}
        sys.sleep(3)
    end


    collectgarbage()
end

-- Save predictions
if arg[1] == 'predict' then
    local predFile = hdf5.open(arg[3]..'/POSE/test-pose.h5', 'w')
    predFile:write('preds', preds)
    predFile:write('scores',scores)
    predFile:close()
elseif arg[1] == 'demo' then
    w.window:close()
end
