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
    -- For managing coordinate transformations between the original image space
    -- and the heatmap

    local pt_ = torch.ones(3)
    pt_[1] = pt[1]
    pt_[2] = pt[2]
    local t = getTransform(center, scale, rot, res)
    if invert then
        t = torch.inverse(t)
    end
    local new_point = (t*pt_):sub(1,2):int()
    return new_point
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

    local len,lenH,lenW
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

function transformBoxAugment(pt1, pt2, outputRes)
    local len = math.max(pt2[2]-pt1[2], pt2[1]-pt1[1])

    local center = torch.zeros(2)
    center[1] = (outputRes+1)/2
    center[2] = (outputRes+1)/2

    local new_pt1 = torch.zeros(2)
    local new_pt2 = torch.zeros(2)

    new_pt1[1] = math.floor(math.max(1,(center[1]-len/2)))
    new_pt1[2] = math.floor(math.max(1,(center[2]-len/2)))
    new_pt2[1] = math.floor(math.max(1,(center[1]+len/2)))
    new_pt2[2] = math.floor(math.max(1,(center[2]+len/2)))

    return new_pt1:int(), new_pt2:int()
end

function getHeatmaps(imHeight, imWidth, ul, br, outputRes, hm)
    local image = require('image')
    -- Crop function tailored to the needs of our system. Provide a center
    -- and scale value and the image will be cropped and resized to the output
    -- resolution determined by res. 'rot' will also rotate the image as needed.

    local newDim,newImg,ht,wd

    newDim = torch.IntTensor({3, br[2] - ul[2], br[1] - ul[1]})
    ht = imHeight
    wd = imWidth

    local newX = torch.Tensor({math.max(1, -ul[1]+1), math.min(br[1], wd) - ul[1]})
    local newY = torch.Tensor({math.max(1, -ul[2]+1), math.min(br[2], ht) - ul[2]})
    local oldX = torch.Tensor({math.max(1, ul[1]+1), math.min(br[1], wd)})
    local oldY = torch.Tensor({math.max(1, ul[2]+1), math.min(br[2], ht)})


    -- mapping
    local newHm = torch.zeros(hm:size(1), ht, wd)
    hm = image.scale(hm:float(), newDim[3], newDim[2])

    newHm:sub(1, hm:size(1), oldY[1],oldY[2],oldX[1],oldX[2]):copy(hm:sub(1,hm:size(1),newY[1],newY[2],newX[1],newX[2]))

    -- Display heatmaps
    if false then
      local colorHms = {}
      for i = 1,16 do 
          colorHms[i] = colorHM(newHm[i])
          colorHms[i]:mul(.7):add(inp)
          w = image.display{image=colorHms[i],win=w}
          sys.sleep(2)
      end
    end

    return newHm
end

-------------------------------------------------------------------------------
-- Cropping
-------------------------------------------------------------------------------

function crop(img, center, scale, rot, res)
    -- Crop function tailored to the needs of our system. Provide a center
    -- and scale value and the image will be cropped and resized to the output
    -- resolution determined by res. 'rot' will also rotate the image as needed.

    local ul = transform({1,1}, center, scale, 0, res, true)
    local br = transform({res,res}, center, scale, 0, res, true)

    local pad = math.floor(torch.norm((ul - br):float())/2 - (br[1]-ul[1])/2)
    if rot ~= 0 then
        ul = ul - pad
        br = br + pad
    end

    local newDim,newImg,ht,wd

    if img:size():size() > 2 then
        newDim = torch.IntTensor({img:size()[1], br[2] - ul[2], br[1] - ul[1]})
        newImg = torch.zeros(newDim[1],newDim[2],newDim[3])
        ht = img:size()[2]
        wd = img:size()[3]
    else
        newDim = torch.IntTensor({br[2] - ul[2], br[1] - ul[1]})
        newImg = torch.zeros(newDim[1],newDim[2])
        ht = img:size()[1]
        wd = img:size()[2]
    end

    local newX = torch.Tensor({math.max(1, -ul[1]+1), math.min(br[1], wd) - ul[1]})
    local newY = torch.Tensor({math.max(1, -ul[2]+1), math.min(br[2], ht) - ul[2]})
    local oldX = torch.Tensor({math.max(1, ul[1]+1), math.min(br[1], wd)})
    local oldY = torch.Tensor({math.max(1, ul[2]+1), math.min(br[2], ht)})

    if newDim:size()[1] > 2 then
        newImg:sub(1,newDim[1],newY[1],newY[2],newX[1],newX[2]):copy(img:sub(1,newDim[1],oldY[1],oldY[2],oldX[1],oldX[2]))
    else
        newImg:sub(newY[1],newY[2],newX[1],newX[2]):copy(img:sub(oldY[1],oldY[2],oldX[1],oldX[2]))
    end

    if rot ~= 0 then
        newImg = image.rotate(newImg, rot * math.pi / 180, 'bilinear')
        if newDim:size()[1] > 2 then
            newImg = newImg:sub(1,newDim[1],pad,newDim[2]-pad,pad,newDim[3]-pad)
        else
            newImg = newImg:sub(pad,newDim[1]-pad,pad,newDim[2]-pad)
        end
    end

    newImg = image.scale(newImg,res,res)
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
        newImg:sub(1,newDim[1],newY[1],newY[2],newX[1],newX[2]):copy(img:sub(1,newDim[1],oldY[1],oldY[2],oldX[1],oldX[2]))
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

function compileImages(imgs, nrows, ncols, resH, resW)
    -- Assumes the input images are all square/the same resolution
    local totalImg = torch.zeros(3,nrows*resH,ncols*resW)
    for i = 1,#imgs do
        local r = torch.floor((i-1)/ncols) + 1
        local c = ((i - 1) % ncols) + 1
        totalImg:sub(1,3,(r-1)*resH+1,r*resH,(c-1)*resW+1,c*resW):copy(imgs[i])
    end
    return totalImg
end

-------------------------------------------------------------------------------
-- Non-maximum Suppression
-------------------------------------------------------------------------------

-- Set up max network for NMS
nms_window_size = 3
nms_pad = (nms_window_size - 1)/2
maxlayer = nn.Sequential()
if cudnn then
    maxlayer:add(cudnn.SpatialMaxPooling(nms_window_size, nms_window_size,1,1, nms_pad, nms_pad))
    maxlayer:cuda()
else
    maxlayer:add(nn.SpatialMaxPooling(nms_window_size, nms_window_size,1,1, nms_pad,nms_pad))
end
maxlayer:evaluate()

function local_maxes(hm, n, c, s, hm_idx)
    hm = torch.Tensor(1,16,64,64):copy(hm):float()
    if hm_idx then hm = hm:sub(1,-1,hm_idx,hm_idx) end
    local hm_dim = hm:size()
    local max_out
    -- First do nms
    if cudnn then
        local hmCuda = torch.CudaTensor(1, hm_dim[2], hm_dim[3], hm_dim[4])
        hmCuda:copy(hm)
        max_out = maxlayer:forward(hmCuda)
        cutorch.synchronize()
    else
        max_out = maxlayer:forward(hm)
    end

    local nms = torch.cmul(hm, torch.eq(hm, max_out:float()):float())[1]
    -- Loop through each heatmap retrieving top n locations, and their scores
    local pred_coords = torch.Tensor(hm_dim[2], n, 2)
    local pred_scores = torch.Tensor(hm_dim[2], n)
    for i = 1, hm_dim[2] do
        local nms_flat = nms[i]:view(nms[i]:nElement())
        local vals,idxs = torch.sort(nms_flat,1,true)
        for j = 1,n do
            local pt = {idxs[j] % 64, torch.ceil(idxs[j] / 64) }
            pred_coords[i][j] = transform(pt, c, s, 0, 64, true)
            pred_scores[i][j] = vals[j]
        end
    end
    return pred_coords, pred_scores
end

-------------------------------------------------------------------------------
-- Drawing functions
-------------------------------------------------------------------------------

function drawGaussian(img, pt, sigma)
    -- Draw a 2D gaussian
    -- Check that any part of the gaussian is in-bounds
    local ul = {math.floor(pt[1] - 3 * sigma), math.floor(pt[2] - 3 * sigma)}
    local br = {math.floor(pt[1] + 3 * sigma), math.floor(pt[2] + 3 * sigma)}
    -- If not, return the image as is
    if (ul[1] > img:size(2) or ul[2] > img:size(1) or br[1] < 1 or br[2] < 1) then return img end
    -- Generate gaussian
    local size = 6 * sigma + 1
    local g = image.gaussian(size) -- , 1 / size, 1)
    -- Usable gaussian range
    local g_x = {math.max(1, -ul[1]), math.min(br[1], img:size(2)) - math.max(1, ul[1]) + math.max(1, -ul[1])}
    local g_y = {math.max(1, -ul[2]), math.min(br[2], img:size(1)) - math.max(1, ul[2]) + math.max(1, -ul[2])}
    -- Image range
    local img_x = {math.max(1, ul[1]), math.min(br[1], img:size(2))}
    local img_y = {math.max(1, ul[2]), math.min(br[2], img:size(1))}
    assert(g_x[1] > 0 and g_y[1] > 0)
    img:sub(img_y[1], img_y[2], img_x[1], img_x[2]):add(g:sub(g_y[1], g_y[2], g_x[1], g_x[2]))
    img[img:gt(1)] = 1
    return img
end

function drawLine(img,pt1,pt2,width,color)
    -- I'm sure there's a line drawing function somewhere in Torch,
    -- but since I couldn't find it here's my basic implementation
    local color = color or {1,1,1}
    local m = torch.dist(pt1:double(),pt2:double())
    local dy = (pt2[2] - pt1[2])/m
    local dx = (pt2[1] - pt1[1])/m
    for j = 1,width do
        local start_pt1 = torch.Tensor({pt1[1] + (-width/2 + j-1)*dy, pt1[2] - (-width/2 + j-1)*dx})
        start_pt1:ceil()
        for i = 1,torch.ceil(m) do
            local y_idx = torch.ceil(start_pt1[2]+dy*i)
            local x_idx = torch.ceil(start_pt1[1]+dx*i)
            if y_idx - 1 > 0 and x_idx -1 > 0 and y_idx < img:size(2) and x_idx < img:size(3) then
                img:sub(1,1,y_idx-1,y_idx,x_idx-1,x_idx):fill(color[1])
                img:sub(2,2,y_idx-1,y_idx,x_idx-1,x_idx):fill(color[2])
                img:sub(3,3,y_idx-1,y_idx,x_idx-1,x_idx):fill(color[3])
            end
        end 
    end
    img[img:gt(1)] = 1

    return img
end

function colorHM(x)
    -- Converts a one-channel grayscale image to a color heatmap image
    local function gauss(x,a,b,c)
        return torch.exp(-torch.pow(torch.add(x,-b),2):div(2*c*c)):mul(a)
    end
    local cl = torch.zeros(3,x:size(1),x:size(2))
    cl[1] = gauss(x,.5,.6,.2) + gauss(x,1,.8,.3)
    cl[2] = gauss(x,1,.5,.3)
    cl[3] = gauss(x,1,.2,.3)
    cl[cl:gt(1)] = 1
    return cl
end


-------------------------------------------------------------------------------
-- Flipping functions
-------------------------------------------------------------------------------

function shuffleLRMPII(x)
    local dim
    if x:nDimension() == 4 then
        dim = 2
    else
        assert(x:nDimension() == 3)
        dim = 1
    end

    local matched_parts = {
        {1,6},   {2,5},   {3,4},
        {11,16}, {12,15}, {13,14}
    }

    for i = 1,#matched_parts do
        local idx1, idx2 = unpack(matched_parts[i])
        local tmp = x:narrow(dim, idx1, 1):clone()
        x:narrow(dim, idx1, 1):copy(x:narrow(dim, idx2, 1))
        x:narrow(dim, idx2, 1):copy(tmp)
    end

    return x
end

function shuffleLRCOCO(x)
    local dim
    if x:nDimension() == 4 then
        dim = 2
    else
        assert(x:nDimension() == 3)
        dim = 1
    end

    local matched_parts = {{2,3},   {4,5},   {6,7},
                    {8,9}, {10,11}, {12,13},
                    {14,15}, {16,17}}


    for i = 1,#matched_parts do
        local idx1, idx2 = unpack(matched_parts[i])
        local tmp = x:narrow(dim, idx1, 1):clone()
        x:narrow(dim, idx1, 1):copy(x:narrow(dim, idx2, 1))
        x:narrow(dim, idx2, 1):copy(tmp)
    end

    return x
end

function shuffleLRAIC(x)
    local dim
    if x:nDimension() == 4 then
        dim = 2
    else
        assert(x:nDimension() == 3)
        dim = 1
    end

    local matched_parts = {{1,6},   {2,5},   {3,4},
                        {11,16}, {12,15}, {13,14}}


    for i = 1,#matched_parts do
        local idx1, idx2 = unpack(matched_parts[i])
        local tmp = x:narrow(dim, idx1, 1):clone()
        x:narrow(dim, idx1, 1):copy(x:narrow(dim, idx2, 1))
        x:narrow(dim, idx2, 1):copy(tmp)
    end

    return x
end

function ConvertMpii2Aic(hmMpii)
    local dim
    local hmAic
    if hmMpii:nDimension() == 4 then
        dim = 2
        hmAic = torch.zeros(hmMpii:size(1),14,hmMpii:size(3),hmMpii:size(4))
    else
        assert(hmMpii:nDimension() == 3)
        dim = 1
        hmAic = torch.zeros(14,hmMpii:size(2),hmMpii:size(3))
    end
    local mpii2aic = {13,12,11,14,15,16,3,2,1,4,5,6,10,9}
    for i = 1,14 do
        hmAic:narrow(dim,i,1):copy(hmMpii:narrow(dim,mpii2aic[i],1))
    end
    return hmAic
end

function flip(x)
    require 'image'
    local y = torch.FloatTensor(x:size())
    for i = 1, x:size(1) do
        image.hflip(y[i], x[i]:float())
    end
    return y:typeAs(x)
end

function grey(x)
    local y = torch.FloatTensor(x:size())
    for i = 1,3 do
        y[1][i] = 0.3*x:float()[1][1] + 0.59*x:float()[1][2] + 0.11*x:float()[1][3]
    end
    return y:typeAs(x)
end

