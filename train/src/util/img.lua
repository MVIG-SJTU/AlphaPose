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

-------------------------------------------------------------------------------
-- Coordinate transformation
-------------------------------------------------------------------------------

function getTransform(center, scale, rot, res)
    local h = 200 * scale
    local t = torch.eye(3)

    -- Scaling
    t[1][1] = res / h
    t[2][2] = res / h

    -- Translation
    t[1][3] = res * (-center[1] / h + .5)
    t[2][3] = res * (-center[2] / h + .5)

    -- Rotation
    if rot ~= 0 then
        rot = -rot
        local r = torch.eye(3)
        local ang = rot * math.pi / 180
        local s = math.sin(ang)
        local c = math.cos(ang)
        r[1][1] = c
        r[1][2] = -s
        r[2][1] = s
        r[2][2] = c
        -- Need to make sure rotation is around center
        local t_ = torch.eye(3)
        t_[1][3] = -res/2
        t_[2][3] = -res/2
        local t_inv = torch.eye(3)
        t_inv[1][3] = res/2
        t_inv[2][3] = res/2
        t = t_inv * r * t_ * t
    end

    return t
end

function transform(pt, center, scale, rot, res, invert)
    local pt_ = torch.ones(3)
    pt_[1],pt_[2] = pt[1]-1,pt[2]-1

    local t = getTransform(center, scale, rot, res)
    if invert then
        t = torch.inverse(t)
    end
    local new_point = (t*pt_):sub(1,2)

    return new_point:int():add(1)
end

function transformBox(pt, ul, br, inpH, inpW, resH, resW)
    local center = torch.zeros(2)
    center[1] = (br[1]-ul[1])/2
    center[2] = (br[2]-ul[2])/2
    
    local len,lenH,lenW
    len = math.max(br[2] - ul[2], (br[1] - ul[1])*inpH/inpW)
    lenH = len
    lenW = len*inpW/inpH
    local _pt = torch.zeros(2)
    _pt[1] = pt[1]-ul[1]
    _pt[2] = pt[2]-ul[2]

    --Move to center
    _pt[1] = _pt[1]+math.max(0,(lenW/2 - center[1]))
    _pt[2] = _pt[2]+math.max(0,(lenH/2 - center[2]))

    local new_point = (_pt*resH)/lenH+1
    return new_point:int()
end

function transformBoxInvert(pt, ul, br, inpH, inpW, resH, resW)
    local center = torch.zeros(2)
    center[1] = (br[1]-ul[1])/2
    center[2] = (br[2]-ul[2])/2

    local len,lenH,lenW,res
    len = math.max(br[2] - ul[2], (br[1] - ul[1])*inpH/inpW)
    lenH = len
    lenW = len*inpW/inpH
    local _pt = (pt*lenH)/resH

    _pt[1] = _pt[1]-math.max(0,(lenW/2 - center[1]))
    _pt[2] = _pt[2]-math.max(0,(lenH/2 - center[2]))

    local new_point = torch.zeros(2)
    new_point[1] = _pt[1]+ul[1]-1
    new_point[2] = _pt[2]+ul[2]-1
    return new_point:int():add(1)
end

function transformPreds(coords, center, scale, res)
    local origDims = coords:size()
    coords = coords:view(-1,2)
    local newCoords = coords:clone()
    for i = 1,coords:size(1) do
        newCoords[i] = transform(coords[i], center, scale, 0, res, 1)
    end
    return newCoords:view(origDims)
end

function transformBoxPreds(coords, ul, br, inpH, inpW, resH, resW)
    local origDims = coords:size()
    coords = coords:view(-1,2)
    local newCoords = coords:clone()
    for i = 1,coords:size(1) do
        newCoords[i] = transformBoxInvert(coords[i], ul, br, inpH, inpW, resH, resW)
    end
    return newCoords:view(origDims)
end

-------------------------------------------------------------------------------
-- Cropping
-------------------------------------------------------------------------------

function checkDims(dims)
    return dims[3] < dims[4] and dims[5] < dims[6]
end

function crop(img, center, scale, rot, res)
    local ndim = img:nDimension()
    if ndim == 2 then img = img:view(1,img:size(1),img:size(2)) end
    local ht,wd = img:size(2), img:size(3)
    local tmpImg,newImg = img, torch.zeros(img:size(1), res, res)

    -- Modify crop approach depending on whether we zoom in/out
    -- This is for efficiency in extreme scaling cases
    local scaleFactor = (200 * scale) / res
    if scaleFactor < 2 then scaleFactor = 1
    else
        local newSize = math.floor(math.max(ht,wd) / scaleFactor)
        if newSize < 2 then
           -- Zoomed out so much that the image is now a single pixel or less
           if ndim == 2 then newImg = newImg:view(newImg:size(2),newImg:size(3)) end
           return newImg
        else
           tmpImg = image.scale(img,newSize)
           ht,wd = tmpImg:size(2),tmpImg:size(3)
        end
    end

    -- Calculate upper left and bottom right coordinates defining crop region
    local c,s = center:float()/scaleFactor, scale/scaleFactor
    local ul = transform({1,1}, c, s, 0, res, true)
    local br = transform({res+1,res+1}, c, s, 0, res, true)
    if scaleFactor >= 2 then br:add(-(br - ul - res)) end

    -- If the image is to be rotated, pad the cropped area
    local pad = math.ceil(torch.norm((ul - br):float())/2 - (br[1]-ul[1])/2)
    if rot ~= 0 then ul:add(-pad); br:add(pad) end

    -- Define the range of pixels to take from the old image
    local old_ = {1,-1,math.max(1, ul[2]), math.min(br[2], ht+1) - 1,
                       math.max(1, ul[1]), math.min(br[1], wd+1) - 1}
    -- And where to put them in the new image
    local new_ = {1,-1,math.max(1, -ul[2] + 2), math.min(br[2], ht+1) - ul[2],
                       math.max(1, -ul[1] + 2), math.min(br[1], wd+1) - ul[1]}

    -- Initialize new image and copy pixels over
    local newImg = torch.zeros(img:size(1), br[2] - ul[2], br[1] - ul[1])
    if not pcall(function() newImg:sub(unpack(new_)):copy(tmpImg:sub(unpack(old_))) end) then
       print("Error occurred during crop!")
    end

    if rot ~= 0 then
        -- Rotate the image and remove padded area
        newImg = image.rotate(newImg, rot * math.pi / 180, 'bilinear')
        newImg = newImg:sub(1,-1,pad+1,newImg:size(2)-pad,pad+1,newImg:size(3)-pad):clone()
    end

    if scaleFactor < 2 then newImg = image.scale(newImg,res,res) end
    if ndim == 2 then newImg = newImg:view(newImg:size(2),newImg:size(3)) end
    return newImg
end

function cropBox(img, ul, br, rot, resH, resW)
    local pad = math.ceil(torch.norm((ul - br):float())/2 - (br[1]-ul[1])/2)
    if rot ~= 0 then
        ul = ul - pad
        br = br + pad
    end

    local newDim,newImg,len,ht,wd,lenH,lenW
    len = math.max(br[2] - ul[2], (br[1] - ul[1])*resH/resW)
    lenH = len
    lenW = len*resW/resH
    if img:size():size() > 2 then
        newDim = torch.IntTensor({img:size(1), br[2] - ul[2], br[1] - ul[1]})
        newImg = torch.zeros(newDim[1],lenH,lenW)
        ht = img:size(2)
        wd = img:size(3)
    else
        newDim = torch.IntTensor({br[2] - ul[2], br[1] - ul[1]})
        newImg = torch.zeros(lenH,lenW)
        ht = img:size(1)
        wd = img:size(2)
    end

    local newX = torch.Tensor({math.max(1, -ul[1] + 2), math.min(br[1], wd+1) - ul[1]})
    local newY = torch.Tensor({math.max(1, -ul[2] + 2), math.min(br[2], ht+1) - ul[2]})
    local newCenter = torch.Tensor({(newX[1]+newX[2])/2,(newY[1]+newY[2])/2})
    --Move to center
    newX[1] = newX[1]+math.floor(math.max(0,(lenW/2 - newCenter[1])))
    newY[1] = newY[1]+math.floor(math.max(0,(lenH/2 - newCenter[2])))
    newX[2] = newX[2]+math.floor(math.max(0,(lenW/2 - newCenter[1])))
    newY[2] = newY[2]+math.floor(math.max(0,(lenH/2 - newCenter[2])))

    local oldX = torch.Tensor({math.max(1, ul[1]), math.min(br[1], wd+1) - 1})
    local oldY = torch.Tensor({math.max(1, ul[2]), math.min(br[2], ht+1) - 1})
    if newDim:size(1) > 2 then
        if pcall(function () 
                newImg:sub(1,newDim[1],newY[1],newY[2],newX[1],newX[2])
                :copy(img:sub(1,newDim[1],oldY[1],oldY[2],oldX[1],oldX[2]))
                end) then 
        else 
            print('newX')
            print(newX)
            print('newY')
            print(newY)
            print('len')
            print(len)
            print('oldX')
            print(oldX)
            print('oldY')
            print(oldY)
            print('wd,ht')
            print(wd,ht)
        end
    else
        newImg:sub(newY[1],newY[2],newX[1],newX[2]):copy(img:sub(oldY[1],oldY[2],oldX[1],oldX[2]))
    end


    if rot ~= 0 then
        newImg = image.rotate(newImg, rot * math.pi / 180, 'bilinear')
        if newDim:size(1) > 2 then
            newImg = newImg:sub(1,newDim[1],pad,newDim[2]-pad,pad,newDim[3]-pad)
        else
            newImg = newImg:sub(pad,newDim[1]-pad,pad,newDim[2]-pad)
        end
    end

    newImg = image.scale(newImg,resW,resH)
    return newImg
end

function twoPointCrop(img, s, pt1, pt2, pad, res)
    local center = (pt1 + pt2) / 2
    local scale = math.max(20*s,torch.norm(pt1 - pt2)) * .007
    scale = scale * pad
    local angle = math.atan2(pt2[2]-pt1[2],pt2[1]-pt1[1]) * 180 / math.pi - 90
    return crop(img, center, scale, angle, res)
end

-------------------------------------------------------------------------------
-- Non-maximum Suppression
-------------------------------------------------------------------------------

function localMaxes(hm, n, c, s, hmIdx, nmsWindowSize)
    -- Set up max network for NMS
    local nmsWindowSize = nmsWindowSize or 3
    local nmsPad = (nmsWindowSize - 1)/2
    local maxlayer = nn.Sequential()
    if cudnn then
        maxlayer:add(cudnn.SpatialMaxPooling(nmsWindowSize, nmsWindowSize,1,1, nmsPad, nmsPad))
        maxlayer:cuda()
    else
        maxlayer:add(nn.SpatialMaxPooling(nmsWindowSize, nmsWindowSize,1,1, nmsPad,nmsPad))
        maxlayer:float()
    end
    maxlayer:evaluate()

    local hmSize = torch.totable(hm:size())
    hm = torch.Tensor(1,unpack(hmSize)):copy(hm):float()
    if hmIdx then hm = hm:sub(1,-1,hmIdx,hmIdx) end
    local hmDim = hm:size()
    local max_out
    -- First do nms
    if cudnn then
        max_out = maxlayer:forward(hm:cuda())
        cutorch.synchronize()
    else
        max_out = maxlayer:forward(hm)
    end

    local nms = torch.cmul(hm, torch.eq(hm, max_out:float()):float())[1]
    -- Loop through each heatmap retrieving top n locations, and their scores
    local predCoords = torch.Tensor(hmDim[2], n, 2)
    local predScores = torch.Tensor(hmDim[2], n)
    for i = 1, hmDim[2] do
        local nms_flat = nms[i]:view(nms[i]:nElement())
        local vals,idxs = torch.sort(nms_flat,1,true)
        for j = 1,n do
            local pt = {(idxs[j]-1) % hmSize[3] + 1, math.floor((idxs[j]-1) / hmSize[3]) + 1 }
            if c then
                predCoords[i][j] = transform(pt, c, s, 0, hmSize[#hmSize], true)
            else
                predCoords[i][j] = torch.Tensor(pt)
            end
            predScores[i][j] = vals[j]
        end
    end
    return predCoords, predScores
end

-------------------------------------------------------------------------------
-- Draw gaussian
-------------------------------------------------------------------------------

function drawGaussian(img, pt, sigma)
    -- Draw a 2D gaussian
    -- Check that any part of the gaussian is in-bounds
    local tmpSize = math.ceil(3*sigma)
    local ul = {math.floor(pt[1] - tmpSize), math.floor(pt[2] - tmpSize)}
    local br = {math.floor(pt[1] + tmpSize), math.floor(pt[2] + tmpSize)}
    -- If not, return the image as is
    if (ul[1] > img:size(2) or ul[2] > img:size(1) or br[1] < 1 or br[2] < 1) then return img end
    -- Generate gaussian
    local size = 2*tmpSize + 1
    local g = image.gaussian(size)
    -- Usable gaussian range
    local g_x = {math.max(1, -ul[1]), math.min(br[1], img:size(2)) - math.max(1, ul[1]) + math.max(1, -ul[1])}
    local g_y = {math.max(1, -ul[2]), math.min(br[2], img:size(1)) - math.max(1, ul[2]) + math.max(1, -ul[2])}
    -- Image range
    local img_x = {math.max(1, ul[1]), math.min(br[1], img:size(2))}
    local img_y = {math.max(1, ul[2]), math.min(br[2], img:size(1))}
    assert(g_x[1] > 0 and g_y[1] > 0)
    img:sub(img_y[1], img_y[2], img_x[1], img_x[2]):cmax(g:sub(g_y[1], g_y[2], g_x[1], g_x[2]))
    return img
end

function drawCircle(img, pt, sigma)
    -- Draw a 2D gaussian
    -- Check that any part of the gaussian is in-bounds
    local tmpSize = math.ceil(3*sigma)
    local ul = {math.floor(pt[1] - tmpSize), math.floor(pt[2] - tmpSize)}
    local br = {math.floor(pt[1] + tmpSize), math.floor(pt[2] + tmpSize)}
    -- If not, return the image as is
    if (ul[1] > img:size(2) or ul[2] > img:size(1) or br[1] < 1 or br[2] < 1) then return img end
    -- Generate gaussian
    local size = 2*tmpSize + 1
    local g = image.gaussian(size)
    -- Usable gaussian range
    local g_x = {math.max(1, -ul[1]), math.min(br[1], img:size(2)) - math.max(1, ul[1]) + math.max(1, -ul[1])}
    local g_y = {math.max(1, -ul[2]), math.min(br[2], img:size(1)) - math.max(1, ul[2]) + math.max(1, -ul[2])}
    -- Image range
    local img_x = {math.max(1, ul[1]), math.min(br[1], img:size(2))}
    local img_y = {math.max(1, ul[2]), math.min(br[2], img:size(1))}
    assert(g_x[1] > 0 and g_y[1] > 0)
    img:sub(img_y[1], img_y[2], img_x[1], img_x[2]):cmax(g:sub(g_y[1], g_y[2], g_x[1], g_x[2]))
    img[img:gt(0)]=1
    return img
end

function drawLine(img, pt1, pt2, width, color)
    if img:nDimension() == 2 then img = img:view(1,img:size(1),img:size(2)) end
    local nChannels = img:size(1)
    color = color or torch.ones(nChannels)
    if type(pt1) == 'table' then pt1 = torch.Tensor(pt1) end
    if type(pt2) == 'table' then pt2 = torch.Tensor(pt2) end

    m = pt1:dist(pt2)
    dy = (pt2[2] - pt1[2])/m
    dx = (pt2[1] - pt1[1])/m
    for j = 1,width do
        start_pt1 = torch.Tensor({pt1[1] + (-width/2 + j-1)*dy, pt1[2] - (-width/2 + j-1)*dx})
        start_pt1:ceil()
        for i = 1,torch.ceil(m) do
            y_idx = torch.ceil(start_pt1[2]+dy*i)
            x_idx = torch.ceil(start_pt1[1]+dx*i)
            if y_idx - 1 > 0 and x_idx -1 > 0 
            and y_idx < img:size(2) and x_idx < img:size(3) then
                for j = 1,nChannels do img[j]:sub(y_idx-1,y_idx,x_idx-1,x_idx):fill(color[j]) end
            end
        end
    end
end

function drawSkeleton(img, preds, scores)
    preds = preds:clone():add(1) -- Account for 1-indexing in lua
    local pairRef = dataset.skeletonRef
    local actThresh = 0.05
    -- Loop through adjacent joint pairings
    for i = 1,#pairRef do
        if scores[pairRef[i][1]] > actThresh and scores[pairRef[i][2]] > actThresh then
            -- Set appropriate line color
            local color
            if pairRef[i][3] == 1 then color = {0,.3,1}
            elseif pairRef[i][3] == 2 then color = {1,.3,0}
            elseif pairRef[i][3] == 3 then color = {0,0,1}
            elseif pairRef[i][3] == 4 then color = {1,0,0}
            else color = {.7,0,.7} end
            -- Draw line
            drawLine(img, preds[pairRef[i][1]], preds[pairRef[i][2]], 4, color, 0)
        end
    end
    return img
end

function heatmapVisualization(set, idx, pred, inp, gt)
    local set = set or 'valid'
    local hmImg
    local tmpInp,tmpHm
    if not inp then
        inp, gt = loadData(set,{idx})
        inp = inp[1]
        gt = gt[1][1]
        tmpInp,tmpHm = inp,gt
    else
        tmpInp = inp
        tmpHm = gt or pred
    end
    local nOut,res = tmpHm:size(1),tmpHm:size(3)
    -- Repeat input image, and darken it to overlay heatmaps
    tmpInp = image.scale(tmpInp,res):mul(.3)
    tmpInp[1][1][1] = 1
    hmImg = tmpInp:repeatTensor(nOut,1,1,1)
    if gt then -- Copy ground truth heatmaps to red channel
        hmImg:sub(1,-1,1,1):add(gt:clone():mul(.7))
    end
    if pred then -- Copy predicted heatmaps to blue channel
        hmImg:sub(1,-1,3,3):add(pred:clone():mul(.7))
    end
    -- Rescale so it is a little easier to see
    hmImg = image.scale(hmImg:view(nOut*3,res,res),256):view(nOut,3,256,256)
    return hmImg, inp
end

-------------------------------------------------------------------------------
-- Flipping functions
-------------------------------------------------------------------------------

function shuffleLR(x)
    local dim
    local matchedParts = dataset.flipRef
    if x:nDimension() == 4 or x:nDimension() == 2 then
        dim = 2
    else
        assert(x:nDimension() == 3 or x:nDimension() == 1)
        dim = 1
    end

    for i = 1,#matchedParts do
        local idx1, idx2 = unpack(matchedParts[i])
        local tmp = x:narrow(dim, idx1, 1):clone()
        x:narrow(dim, idx1, 1):copy(x:narrow(dim, idx2, 1))
        x:narrow(dim, idx2, 1):copy(tmp)
    end
    return x
end

function flip(x)
    require 'image'
    local y = torch.FloatTensor(x:size())
    for i = 1, x:size(1) do
        image.hflip(y[i], x[i]:float())
    end
    return y:typeAs(x)
end
