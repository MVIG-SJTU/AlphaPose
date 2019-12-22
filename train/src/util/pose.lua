-- Update dimension references to account for intermediate supervision
ref.predDim = {dataset.nJoints,5}
ref.outputDim = {}
criterion = nn.ParallelCriterion()
if opt.addParallelSPPE then
  for i = 1,opt.nStack+opt.nParaStack do
      ref.outputDim[i] = {dataset.nJoints, opt.outputResH, opt.outputResW}
      criterion:add(nn[opt.crit .. 'Criterion']())
  end
else
  for i = 1,opt.nStack do
    ref.outputDim[i] = {dataset.nJoints, opt.outputResH, opt.outputResW}
    criterion:add(nn[opt.crit .. 'Criterion']())
  end
end

-- Function for data augmentation, randomly samples on a normal distribution
local function rnd(x) return math.max(-2*x,math.min(2*x,torch.randn(1)[1]*x)) end

-- Code to generate training samples from raw images, center scale format
local function generateSampleCS(set, idx)
    local img = dataset:loadImage(idx)
    local pts, c, s = dataset:getPartInfo(idx)
    local r = 0

    if set == 'train' then
        -- Scale and rotation augmentation
        s = s * (2 ^ rnd(opt.scale))
        r = rnd(opt.rotate)
        if torch.uniform() <= .6 then r = 0 end
    end

    local inp = crop(img, c, s, r, opt.inputResH, opt.inputResH)
    local out = torch.zeros(dataset.nJoints, opt.outputResH, opt.outputResW)
    for i = 1,dataset.nJoints do
        if pts[i][1] > 1 then -- Checks that there is a ground truth annotation
            drawGaussian(out[i], transform(pts[i], c, s, r, opt.outputResH, opt.outputResW), opt.hmGauss)
        end
    end

    if set == 'train' then
        -- Flipping and color augmentation
        if torch.uniform() < .5 then
            inp = flip(inp)
            out = shuffleLR(flip(out))
        end
        inp[1]:mul(torch.uniform(0.6,1.4)):clamp(0,1)
        inp[2]:mul(torch.uniform(0.6,1.4)):clamp(0,1)
        inp[3]:mul(torch.uniform(0.6,1.4)):clamp(0,1)
    end

    return inp,out
end

-- Code to generate training samples from raw images, bbox format
local function generateSampleBox(set, idx)
    local img = dataset:loadImage(idx)
    if set == 'train' then
        -- Color augmentation
        img[1]:mul(torch.uniform(0.7,1.3)):clamp(0,1)
        img[2]:mul(torch.uniform(0.7,1.3)):clamp(0,1)
        img[3]:mul(torch.uniform(0.7,1.3)):clamp(0,1)
    end

    img[1]:add(-0.406)
    img[2]:add(-0.457)
    img[3]:add(-0.480)
    local pts, bndbox, imgset = dataset:getPartInfo(idx)
    --print(imgset)
    local upLeft = torch.Tensor({bndbox[1][1],bndbox[1][2]})
    local bottomRight = torch.Tensor({bndbox[1][3],bndbox[1][4]})
    local ht = bottomRight[2]-upLeft[2]
    local width = bottomRight[1]-upLeft[1]
    local imght = img:size()[2]
    local imgwidth = img:size()[3]
    local scaleRate = torch.uniform(0.2,0.3)
    upLeft[1] = math.max(1,(upLeft[1] - width*scaleRate/2))
    upLeft[2] = math.max(1,(upLeft[2] - ht*scaleRate/2))
    bottomRight[1] = math.min(imgwidth-1,(bottomRight[1] + width*scaleRate/2))
    bottomRight[2] = math.min(imght-1,(bottomRight[2] + ht*scaleRate/2))

    local out_center = torch.zeros(dataset.nJoints, opt.outputResH, opt.outputResW)
    local xmin,ymin,xmax,ymax

    if (opt.usePGPG == true) then
    -----------------------------------------------
    ------------- Doing random samples ------------
    -----------------------------------------------
        local PatchScale = torch.uniform()
        if PatchScale > 0.85 then
            ratio = ht/width
            -- Cut a patch
            if (width < ht) then
                patchWidth = PatchScale*width
                patchHt = patchWidth*ratio
            else
                patchHt = PatchScale*ht
                patchWidth = patchHt/ratio
            end
            xmin = upLeft[1]+torch.uniform()*(width-patchWidth)
            ymin = upLeft[2]+torch.uniform()*(ht-patchHt)
            xmax = xmin+patchWidth+1
            ymax = ymin+patchHt+1
        else
            --Gaussian distribution
            xmin = math.max(1, math.min(upLeft[1]+torch.normal(-0.0142,0.1158)*width, imgwidth-3))
            ymin = math.max(1, math.min(upLeft[2]+torch.normal(0.0043,0.068)*ht,imght-3))
            xmax = math.min(math.max(xmin+2, bottomRight[1]+torch.normal(0.0154,0.1337)*width), imgwidth-3)
            ymax = math.min(math.max(ymin+2, bottomRight[2]+torch.normal(-0.0013,0.0711)*ht), imght-3)
        end

         if opt.addParallelSPPE == true then
            if PatchScale > 0.7 then -- For patch, we 'disable' the parallel SPPE because STN do not need to focus in this case
                for i = 1,dataset.nJoints do
                    if pts[i][1] > 0 and pts[i][1] > upLeft[1] and pts[i][2] > upLeft[2] and pts[i][1] < bottomRight[1] and pts[i][2] < bottomRight[2] then -- Checks that there is a ground truth annotation
                        drawGaussian(out_center[i], transformBox(pts[i],torch.Tensor({xmin,ymin}):int(),torch.Tensor({xmax,ymax}):int(),opt.outputResH,opt.outputResW), opt.hmGauss)
                    end
                end
            else
                for i = 1,dataset.nJoints do
                    if pts[i][1] > 0 and pts[i][1] > upLeft[1] and pts[i][2] > upLeft[2] and pts[i][1] < bottomRight[1] and pts[i][2] < bottomRight[2] then -- Checks that there is a ground truth annotation
                        drawGaussian(out_center[i], transformBox(pts[i],upLeft:int(),bottomRight:int(),opt.outputResH,opt.outputResW), opt.hmGauss)
                    end
                end
            end
        end
        upLeft[1] = xmin; upLeft[2] = ymin;
        bottomRight[1] = xmax; bottomRight[2] = ymax;
    end

    local jointNum = 0
    if imgset == 'coco' or imgset == 'neg' then
        for i = 1,dataset.nJoints_coco do
            if pts[i][1] > 0 and pts[i][1] > upLeft[1] and pts[i][2] > upLeft[2] and pts[i][1] < bottomRight[1] and pts[i][2] < bottomRight[2] then -- Checks that there is a ground truth annotation
                jointNum = jointNum+1
            end
        end
    else
        for i = 1,dataset.nJoints_mpii do
            if pts[i][1] > 0 and pts[i][1] > upLeft[1] and pts[i][2] > upLeft[2] and pts[i][1] < bottomRight[1] and pts[i][2] < bottomRight[2] then -- Checks that there is a ground truth annotation
                jointNum = jointNum+1
            end
        end
    end

    if (opt.usePGPG == true) then
        if jointNum>13 and set == 'train' then
            local switch = torch.uniform()
            if switch >=0.96 then
                bottomRight[1] = (upLeft[1]+bottomRight[1])/2
                bottomRight[2] = (upLeft[2]+bottomRight[2])/2
            elseif switch >=0.92 then
                upLeft[1] = (upLeft[1]+bottomRight[1])/2
                bottomRight[2] = (upLeft[2]+bottomRight[2])/2
            elseif switch >=0.88 then
           	    upLeft[2] = (upLeft[2]+bottomRight[2])/2
               bottomRight[1] = (upLeft[1]+bottomRight[1])/2
            elseif switch >=0.84 then
                upLeft[1] = (upLeft[1]+bottomRight[1])/2
                upLeft[2] = (upLeft[2]+bottomRight[2])/2
            elseif switch >=0.80 then
                bottomRight[1] = (upLeft[1]+bottomRight[1])/2
            elseif switch >=0.76 then
                upLeft[1] = (upLeft[1]+bottomRight[1])/2
            elseif switch >=0.72 then
                bottomRight[2] = (upLeft[2]+bottomRight[2])/2
            elseif switch >=0.68 then
                upLeft[2] = (upLeft[2]+bottomRight[2])/2
            end
        end
    end
    
    local inp = cropBox(img, upLeft:int(),bottomRight:int(), 0, opt.inputResH, opt.inputResW)
    if jointNum==0 and imgset ~= 'neg' then
       inp = torch.zeros(3, opt.inputResH, opt.inputResW)
       -- print("find one person with no annot")
    end
    -----------------------------------
    -- 33 output channels, first 17 channels are coco's, last 16 channels are mpii's
    local out_bigcircle = torch.zeros(dataset.nJoints, opt.outputResH, opt.outputResW)
    local out_smallcircle = torch.zeros(dataset.nJoints, opt.outputResH, opt.outputResW)
    local out = torch.zeros(dataset.nJoints, opt.outputResH, opt.outputResW)
    local setMask = torch.zeros(dataset.nJoints, opt.outputResH, opt.outputResW)
    local ptMask = torch.zeros(dataset.nJoints, 2)
    if imgset == 'coco' then
        for i = 1,dataset.nJoints_coco do
            if pts[i][1] > 0 and pts[i][1] > upLeft[1] and pts[i][2] > upLeft[2] and pts[i][1] < bottomRight[1] and pts[i][2] < bottomRight[2] then -- Checks that there is a ground truth annotation
                drawCircle(out_bigcircle[i], transformBox(pts[i],upLeft:int(),bottomRight:int(),opt.inputResH,opt.inputResW,opt.outputResH,opt.outputResW), opt.hmGauss*2)
                drawCircle(out_smallcircle[i], transformBox(pts[i],upLeft:int(),bottomRight:int(),opt.inputResH,opt.inputResW,opt.outputResH,opt.outputResW), opt.hmGauss)
                drawGaussian(out[i], transformBox(pts[i],upLeft:int(),bottomRight:int(),opt.inputResH,opt.inputResW,opt.outputResH,opt.outputResW), opt.hmGauss)
                ptMask[i]:add(1)
            end
            setMask[i]:add(1)
        end
    elseif imgset == 'mpii' then -- mpii
        for i = 1,dataset.nJoints_mpii do
            if pts[i][1] > 0 and pts[i][1] > upLeft[1] and pts[i][2] > upLeft[2] and pts[i][1] < bottomRight[1] and pts[i][2] < bottomRight[2] then -- Checks that there is a ground truth annotation
                drawCircle(out_bigcircle[i+dataset.nJoints_coco], transformBox(pts[i],upLeft:int(),bottomRight:int(),opt.inputResH,opt.inputResW,opt.outputResH,opt.outputResW), opt.hmGauss*2)
                drawCircle(out_smallcircle[i+dataset.nJoints_coco], transformBox(pts[i],upLeft:int(),bottomRight:int(),opt.inputResH,opt.inputResW,opt.outputResH,opt.outputResW), opt.hmGauss)
                drawGaussian(out[i+dataset.nJoints_coco], transformBox(pts[i],upLeft:int(),bottomRight:int(),opt.inputResH,opt.inputResW,opt.outputResH,opt.outputResW), opt.hmGauss)
            end
            setMask[i+dataset.nJoints_coco]:add(1)
        end
    end

    if set == 'train' then
        -- Flipping and color augmentation
        if torch.uniform() < .5 then
            inp = flip(inp)
            out_bigcircle = shuffleLR(flip(out_bigcircle))
            out_smallcircle = shuffleLR(flip(out_smallcircle))
            out = shuffleLR(flip(out))
            if opt.addParallelSPPE == true then
            	out_center = shuffleLR(flip(out_center))
            end
        end
        -- rotation augmentation
        r = rnd(opt.rotate)
        if torch.uniform() <= .6 then r = 0 end
        if r ~= 0 then
          -- Rotate the image and remove padded area
          inp = image.rotate(inp, r * math.pi / 180, 'bilinear')
          for i = 1,dataset.nJoints do
          	out_bigcircle[i] = image.rotate(out_bigcircle[i], r * math.pi / 180, 'bilinear')
          	out_smallcircle[i] = image.rotate(out_smallcircle[i], r * math.pi / 180, 'bilinear')
            out[i] = image.rotate(out[i], r * math.pi / 180, 'bilinear')
            if opt.addParallelSPPE == true then
                out_center[i] = image.rotate(out_center[i], r * math.pi / 180, 'bilinear')
            end
          end
        end
    end
    -- if imgset == 'neg' then
    --    image.save('pic/'..tostring(idx)..'.jpg',inp)
    -- end
    --print(pts)
    
    if opt.addParallelSPPE == true then
        return inp,out,out_center
    else
        return inp, out_bigcircle, out_smallcircle, out, setMask
    end
end

-- Load in a mini-batch of data
function loadData(set, idxs)
    if type(idxs) == 'table' then idxs = torch.Tensor(idxs) end
    if opt.dataset == 'mpii-cs' then
        generateSample = generateSampleCS
    else
        generateSample = generateSampleBox
    end
    local nsamples = idxs:size(1)
    local input,label,mask

    if opt.addParallelSPPE == true then
        for i = 1,nsamples do
            local tmpInput,tmpLabel,tmpLabelPara
            tmpInput,tmpLabel,tmpLabelPara = generateSample(set, idxs[i])
            tmpInput = tmpInput:view(1,unpack(tmpInput:size():totable()))
            tmpLabel = tmpLabel:view(1,unpack(tmpLabel:size():totable()))
            tmpLabelPara = tmpLabelPara:view(1,unpack(tmpLabelPara:size():totable()))
            if not input then
                input = tmpInput
                label = tmpLabel
                labelPara = tmpLabelPara
            else
                input = input:cat(tmpInput,1)
                label = label:cat(tmpLabel,1)
                labelPara = labelPara:cat(tmpLabelPara,1)
            end
        end

        if opt.nStack > 1 then
            -- Set up label for intermediate supervision
            local newLabel = {}
            for i = 1,opt.nStack do newLabel[i] = label end
            for i = 1,opt.nParaStack do table.insert(newLabel,labelPara) end
            return input,newLabel
        else
            return input,{label,labelPara}
        end
    else
        for i = 1,nsamples do
        local tmpInput,tmpLabel,tmpMask
        tmpInput,tmpLabel1, tmpLabel2, tmpLabel,tmpMask = generateSample(set, idxs[i])
        tmpInput = tmpInput:view(1,unpack(tmpInput:size():totable()))
        tmpLabel1 = tmpLabel1:view(1,unpack(tmpLabel1:size():totable()))
        tmpLabel2 = tmpLabel2:view(1,unpack(tmpLabel2:size():totable()))
        tmpLabel = tmpLabel:view(1,unpack(tmpLabel:size():totable()))
        tmpMask = tmpMask:view(1,unpack(tmpMask:size():totable()))
        if not input then
            input = tmpInput
            label1 = tmpLabel1
            label2 = tmpLabel2
            label = tmpLabel
            mask = tmpMask
        else
            input = input:cat(tmpInput,1)
            label1 = label:cat(tmpLabel1,1)
            label2 = label:cat(tmpLabel2,1)
            label = label:cat(tmpLabel,1)
            mask = mask:cat(tmpMask,1)
        end
        end

        if opt.nStack > 1 then
            -- Set up label for intermediate supervision
            local newLabel = {}
            for i = 1,2 do newLabel[i] = label1 end
            for i = 3,4 do newLabel[i] = label2 end
            for i = 5,8 do newLabel[i] = label end
            return input,newLabel,mask
        else
            return input,label,mask
        end
    end
end

function postprocess(set, idx, output)
    local tmpOutput
    if type(output) == 'table' then tmpOutput = output[#output]
    else tmpOutput = output end
    local p = getPreds(tmpOutput)
    local scores = torch.zeros(p:size(1),p:size(2),1)

    -- Very simple post-processing step to improve performance at tight PCK thresholds
    for i = 1,p:size(1) do
        for j = 1,p:size(2) do
            local hm = tmpOutput[i][j]
            local pX,pY = p[i][j][1], p[i][j][2]
            scores[i][j] = hm[pY][pX]
            if pX > 1 and pX < opt.outputResW and pY > 1 and pY < opt.outputResH then
               local diff = torch.Tensor({hm[pY][pX+1]-hm[pY][pX-1], hm[pY+1][pX]-hm[pY-1][pX]})
               p[i][j]:add(diff:sign():mul(.25))
            end
        end
    end
    p:add(0.5)

    -- Transform predictions back to original coordinate space
    local p_tf = torch.zeros(p:size())
    if opt.dataset == 'mpii-cs' then
        for i = 1,p:size(1) do
            _,c,s = dataset:getPartInfo(idx[i])
            p_tf[i]:copy(transformPreds(p[i], c, s, opt.outputRes))
        end
    else
        for i = 1,p:size(1) do
            local bndbox
            _, bndbox, _ = dataset:getPartInfo(idx[i])
            local upLeft = torch.Tensor({bndbox[1][1],bndbox[1][2]})
            local bottomRight = torch.Tensor({bndbox[1][3],bndbox[1][4]})
            p_tf[i]:copy(transformBoxPreds(p[i], upLeft:int(), bottomRight:int(), opt.inputResH, opt.inputResW, opt.outputResH, opt.outputResW))
        end
    end

    return p_tf:cat(p,3):cat(scores,3)
end

function accuracy(output,label)
    if type(output) == 'table' and opt.addParallelSPPE == true then
        return heatmapAccuracy(output[opt.nStack],label[opt.nStack],nil,dataset.accIdxs)
    elseif type(output) == 'table' then
        return heatmapAccuracy(output[#output],label[#output],nil,dataset.accIdxs)
    else
        return heatmapAccuracy(output,label,nil,dataset.accIdxs)
    end
end

function accuracyCOCO(output,label)
    if type(output) == 'table' and opt.addParallelSPPE == true then
        return heatmapAccuracy(output[opt.nStack],label[opt.nStack],nil,dataset.accIdxs_coco)
    elseif type(output) == 'table' then
        return heatmapAccuracy(output[#output],label[#output],nil,dataset.accIdxs_coco)
    else
        return heatmapAccuracy(output,label,nil,dataset.accIdxs_coco)
    end
end
