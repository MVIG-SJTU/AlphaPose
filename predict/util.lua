require 'torch'
require 'xlua'
require 'nn'
require 'nnx'
require 'nngraph'
require 'image'
require 'hdf5'
require 'sys'

require 'cunn'
require 'cutorch'
require 'cudnn'

function string:split( inSplitPattern, outResults )
  if not outResults then
    outResults = { }
  end
  local theStart = 1
  local theSplitStart, theSplitEnd = string.find( self, inSplitPattern, theStart )
  while theSplitStart do
    table.insert( outResults, string.sub( self, theStart, theSplitStart-1 ) )
    theStart = theSplitEnd + 1
    theSplitStart, theSplitEnd = string.find( self, inSplitPattern, theStart )
  end
  table.insert( outResults, string.sub( self, theStart ) )
  return outResults
end


function loadAnnotations(set)
    -- Load up a set of annotations for either: 'train', 'valid', or 'test'
    -- There is no part information in 'test'

    local a = hdf5.open(set .. '/test-bbox.h5')
    annot = {}

    -- Read in annotation information from hdf5 file
    local tags = {'xmin','ymin','xmax','ymax'}
    for _,tag in ipairs(tags) do annot[tag] = a:read(tag):all() end
    annot.nsamples = annot.xmin:size()[1]
    a:close()

    -- Load in image file names
    -- (workaround for not being able to read the strings in the hdf5 file)
    annot.images = {}
    local toIdxs = {}
    local namesFile = io.open(set .. '/test-images.txt')
    local idx = 1
    for line in namesFile:lines() do
        annot.images[idx] = line
        if not toIdxs[line] then toIdxs[line] = {} end
        table.insert(toIdxs[line], idx)
        idx = idx + 1
    end
    namesFile:close()

    -- Load in index file names
    -- (workaround for not being able to read the strings in the hdf5 file)
    annot.index = {}
    annot.index_num = {}
    local indexFile = io.open(set .. '/index.txt')
    local idx = 0
    for line in indexFile:lines() do
        idx = idx + 1
        line_dict = line:split(" ")
        annot.index[idx] = line_dict[1]
        annot.index_num[idx] = tonumber(line_dict[3])-tonumber(line_dict[2])+1
    end
    annot.nImages = idx
    indexFile:close()

    -- Load in score file names
    -- (workaround for not being able to read the strings in the hdf5 file)
    annot.scores = {}
    local namesFile = io.open(set .. '/score-proposals.txt')
    local idx = 0
    for line in namesFile:lines() do
        idx = idx + 1
        annot.scores[idx] = tonumber(line)
    end
    namesFile:close()

    -- This allows us to reference all people who are in the same image
    annot.imageToIdxs = toIdxs

    return annot
end


function getPreds(hms, pt1, pt2, inpH, inpW, resH, resW)
    -- Get locations of maximum activations
    local max, idx = torch.max(hms:view(hms:size(1), hms:size(2), hms:size(3) * hms:size(4)), 3)
    local preds = torch.zeros(hms:size(1), hms:size(2), 2):float()
    local miniBat = hms:size(1)
    for k = 1, miniBat do
        preds[k]:copy(idx[k]:repeatTensor(1,1,2))
    end
    preds[{{}, {}, 1}]:apply(function(x) return (x - 1) % hms:size(4) + 1 end)
    preds[{{}, {}, 2}]:add(-1):div(hms:size(4)):floor():add(1)
    local predMask = max:gt(0):repeatTensor(1, 1, 2):float()
    preds:add(-1):cmul(predMask):add(1)
    -- Very simple post-processing step to improve performance at tight PCK thresholds
    for i = 1,preds:size(1) do
       for j = 1,preds:size(2) do
            local hm = hms[i][j]
            local pX,pY = preds[i][j][1], preds[i][j][2]
            --scores[i][j] = hm[pY][pX]
            if pX > 1 and pX < resW and pY > 1 and pY < resH then
                local diff = torch.Tensor({hm[pY][pX+1]-hm[pY][pX-1], hm[pY+1][pX]-hm[pY-1][pX]})
                preds[i][j]:add(diff:sign():mul(.25):float())
            end
        end
    end
    preds:add(-0.5) 

    -- Get transformed coordinates
    local preds_tf = torch.zeros(preds:size())


    for i = 1,hms:size(1) do        -- Number of samples
        for j = 1,hms:size(2) do    -- Number of output heatmaps for one sample
            preds_tf[i][j] = transformBoxInvert(preds[i][j],pt1[i],pt2[i],inpH,inpW,resH,resW)
        end
    end

    return preds, preds_tf, max
end

function getPreds4crop(hms, pt1, pt2, inpH, inpW, resH, resW)
    if hms:size():size() == 3 then hms = hms:view(1, hms:size(1), hms:size(2), hms:size(3)) end
    assert(hms:dim() == 4, 'Input must be 4-D tensor')
    -- Get locations of maximum activations
    local max, idx = torch.max(hms:view(hms:size(1), hms:size(2), hms:size(3) * hms:size(4)), 3)
    local preds = torch.zeros(hms:size(1), hms:size(2), 2):float()
    local miniBat = hms:size(1)
    for k = 1, miniBat do
        preds[k]:copy(idx[k]:repeatTensor(1,1,2))
    end
    preds[{{}, {}, 1}]:apply(function(x) return (x - 1) % hms:size(4) + 1 end)
    preds[{{}, {}, 2}]:add(-1):div(hms:size(4)):floor():add(1)
    local predMask = max:gt(0):repeatTensor(1, 1, 2):float()
    preds:add(-1):cmul(predMask):add(1)
    -- Very simple post-processing step to improve performance at tight PCK thresholds
    for i = 1,preds:size(1) do
       for j = 1,preds:size(2) do
            local hm = hms[i][j]
            local pX,pY = preds[i][j][1], preds[i][j][2]
            --scores[i][j] = hm[pY][pX]
            if pX > 1 and pX < resW and pY > 1 and pY < resH then
                local diff = torch.Tensor({hm[pY][pX+1]-hm[pY][pX-1], hm[pY+1][pX]-hm[pY-1][pX]})
                preds[i][j]:add(diff:sign():mul(.25):float())
            end
        end
    end
    preds:add(-0.5)

    -- Get transformed coordinates
    local preds_tf = torch.zeros(preds:size())


    for i = 1,hms:size(1) do        -- Number of samples
        for j = 1,hms:size(2) do    -- Number of output heatmaps for one sample
            preds_tf[i][j] = transformBoxInvert(preds[i][j],pt1,pt2,inpH,inpW,resH,resW)
        end
    end

    return preds, preds_tf, max
end


function getPredsOriIm(hms, len, outres)

    if hms:size():size() == 3 then hms = hms:view(1, hms:size(1), hms:size(2), hms:size(3)) end
    assert(hms:dim() == 4, 'Input must be 4-D tensor')
    -- Get locations of maximum activations
    local max, idx = torch.max(hms:view(hms:size(1), hms:size(2), hms:size(3) * hms:size(4)), 3)
    local preds = torch.repeatTensor(idx, 1, 1, 2):float()
    preds[{{}, {}, 1}]:apply(function(x) return (x - 1) % hms:size(4) + 1 end)
    preds[{{}, {}, 2}]:add(-1):div(hms:size(4)):floor():add(1)
    local predMask = max:gt(0):repeatTensor(1, 1, 2):float()
    preds:add(-1):cmul(predMask):add(1)
    
    -- Very simple post-processing step to improve performance at tight PCK thresholds
    for i = 1,preds:size(1) do
        for j = 1,preds:size(2) do
            local hm = hms[i][j]
            local pX,pY = preds[i][j][1], preds[i][j][2]
            --scores[i][j] = hm[pY][pX]
            if pX > 1 and pX < hms:size(4) and pY > 1 and pY < hms:size(3) then
               local diff = torch.Tensor({hm[pY][pX+1]-hm[pY][pX-1], hm[pY+1][pX]-hm[pY-1][pX]})
               preds[i][j]:add(diff:sign():mul(.25 * len/outres):float())
            end
        end
    end
    preds:add(1 * len/outres)


    return preds-1, max
end

function getImgHM(hms, ul, br, ht, wd)
    if hms:size():size() == 4 then hms = hms:view(hms:size(2), hms:size(3), hms:size(4)) end
    -- local hm_new = torch.zeros(17,64,64)
    -- for j = 1,hms:size(1) do    -- Number of output heatmaps for one sample
    --     hm_new[j]:sub(2,64,2,64):copy(hms[j]:sub(1,63,1,63))
    -- end
    -- -- Get transformed coordinates
    -- hms = hm_new
    local hm_img = torch.zeros(14,ht,wd):double()

    local len
    len = math.max(br[2] - ul[2], br[1] - ul[1])
    hms = image.scale(hms,len,len)
    hms = hms:view(1, hms:size(1), hms:size(2), hms:size(3))
    local newX = torch.Tensor({math.max(1, -ul[1] + 1), math.min(br[1], wd+1) - ul[1]})
    local newY = torch.Tensor({math.max(1, -ul[2] + 1), math.min(br[2], ht+1) - ul[2]})
    local newCenter = torch.Tensor({(newX[1]+newX[2])/2,(newY[1]+newY[2])/2})
    --Move to center
    newX[1] = newX[1]+math.floor(math.max(0,(len/2 - newCenter[1])))
    newY[1] = newY[1]+math.floor(math.max(0,(len/2 - newCenter[2])))
    newX[2] = newX[2]+math.floor(math.max(0,(len/2 - newCenter[1])))
    newY[2] = newY[2]+math.floor(math.max(0,(len/2 - newCenter[2])))

    local oldX = torch.Tensor({math.max(1, ul[1]), math.min(br[1], wd+1)-1})
    local oldY = torch.Tensor({math.max(1, ul[2]), math.min(br[2], ht+1)-1})
    

    for j = 1,hms:size(2) do    -- Number of output heatmaps for one sample
        hm_img[j]:sub(oldY[1],oldY[2],oldX[1],oldX[2]):copy(hms[1][j]:sub(newY[1],newY[2],newX[1],newX[2]))
    end

    return hm_img
end  

function getPredsOrig(hms, imght, imgwidth, len, outres)

    if hms:size():size() == 3 then hms = hms:view(1, hms:size(1), hms:size(2), hms:size(3)) end

    -- Get locations of maximum activations
    local max, idx = torch.max(hms:view(hms:size(1), hms:size(2), hms:size(3) * hms:size(4)), 3)
    local preds = torch.repeatTensor(idx, 1, 1, 2):float()
    preds[{{}, {}, 1}]:apply(function(x) return (x - 1) % hms:size(4) + 1 end)
    preds[{{}, {}, 2}]:add(-1):div(hms:size(4)):floor():add(1)

    -- Very simple post-processing step to improve performance at tight PCK thresholds
    for i = 1,preds:size(1) do
       for j = 1,preds:size(2) do
            local hm = hms[i][j]
            local pX,pY = preds[i][j][1], preds[i][j][2]
            scores[i][j] = hm[pY][pX]
            if pX > 1 and pX < hms:size(4) and pY > 1 and pY < hms:size(3) then
               local diff = torch.Tensor({hm[pY][pX+1]-hm[pY][pX-1], hm[pY+1][pX]-hm[pY-1][pX]})
               preds[i][j]:add(diff:sign():mul(0.25 * len/outres):float())
            end
        end
    end
    -- preds:add(1 * len/outres)

    return preds-1, max
end

-------------------------------------------------------------------------------
-- Functions for hm merge
-------------------------------------------------------------------------------

function tryMatch(hm_img,centroid)
    if hm_img:size():size() == 3 then hm_img = hm_img:view(1, hm_img:size(1), hm_img:size(2), hm_img:size(3)) end
    if centroid:size():size() == 3 then centroid = centroid:view(1, centroid:size(1), centroid:size(2), centroid:size(3)) end
    local insertion = torch.sum(torch.sqrt(torch.cmul(hm_img, centroid)))
    local union = torch.sum(hm_img)+torch.sum(centroid)-insertion
    local hm1 = hm_img:clone()
    local hm2 = centroid:clone()
    for i=1, hm1:size(2) do
        local max, _ = torch.max(hm1[1][i]:view(hm1:size(3) * hm1:size(4)), 1)
        if max[1] < 0.2 then
            hm1[1][i] = torch.zeros(hm1:size(3), hm1:size(4))
        end
        local max, _ = torch.max(hm2[1][i]:view(hm2:size(3) * hm2:size(4)), 1)
        if max[1] < 0.2 then
            hm2[1][i] = torch.zeros(hm2:size(3), hm2:size(4))
        end
    end
    return insertion/union, insertion/torch.sum(hm1), insertion/torch.sum(hm2)
end  

function merge(centroid,hm_img,centroid_pf)
    if hm_img:size():size() == 3 then hm_img = hm_img:view(1, hm_img:size(1), hm_img:size(2), hm_img:size(3)) end
    if centroid:size():size() == 3 then centroid = centroid:view(1, centroid:size(1), centroid:size(2), centroid:size(3)) end
    for i=1, centroid:size(2) do
        local max, _ = torch.max(hm_img[1][i]:view(hm_img:size(3) * hm_img:size(4)), 1)
        if max[1] > 0.5 then
            centroid[1][i] = centroid[1][i]:mul(centroid_pf[i]):add(hm_img[1][i]):div(centroid_pf[i]+1)
            centroid_pf[i] = centroid_pf[i]+1
        else
            centroid[1][i] = centroid[1][i]
            centroid_pf[i] = centroid_pf[i]
        end
    end

    return centroid,centroid_pf
end   
-------------------------------------------------------------------------------
-- Functions for setting up the demo display
-------------------------------------------------------------------------------
function drawSkeletoncoco(input, hms, coords)

    local im = input:clone()
    if hms:size():size() == 4 then hms = hms:view(hms:size(2), hms:size(3), hms:size(4)) end
    local pairRef = {
        {1,2},      {1,3},      {2,3},
        {7,9},      {8,10},      {9,11},
        {1,12},      {1,13},
        {12,14},     {14,16}, 
        {13,15},     {15,17}
    }

    local partNames = {'nose','left eye','right eye','left ear','right ear',
                       'left shoulder','right shoulder','left elbow','right elbow','left wrist','right wrist',
                       'left hip','right hip','left knee','right knee','left ankle','right ankle'}
    local partColor = {1,1,1,2,2,2,0,0,0,0,3,3,3,4,4,4}

    local actThresh = 0.2

    -- Loop through adjacent joint pairings
    for i = 1,#pairRef do
        if hms[pairRef[i][1]]:max() > actThresh and hms[pairRef[i][2]]:max() > actThresh then
            -- Set appropriate line color
            local color
            if partColor[pairRef[i][1]] == 1 then color = {0,.3,1}
            elseif partColor[pairRef[i][1]] == 2 then color = {1,.3,0}
            elseif partColor[pairRef[i][1]] == 3 then color = {0,0,1}
            elseif partColor[pairRef[i][1]] == 4 then color = {1,0,0}
            else color = {.7,0,.7} end

            -- Draw line
            im = drawLine(im, coords[pairRef[i][1]], coords[pairRef[i][2]], 2, color, 0)
        end
    end

    return im
end

function drawSkeletonAic(input, hms, coords)

    local im = input:clone()

    local pairRef = {
        {1,2},  {2,3},
        {4,5},  {5,6},
        {13,14},
        {14,1}, {1,7}, {14,4},  {4,10}, {7,10},
        {8,7},  {8,9},
        {10,11},{11,12}
    }

    local partNames = {'RElb','RWri','LElb','LWri','Head','RSho','RHip','LSho','LHip','Pelv','RLeg','RAnk','LLeg','LAnk'}
    local partColor = {1,1,0,2,2,0,0,3,0,4,4,0,0,0}

    local actThresh = 0.2

    -- Loop through adjacent joint pairings
    for i = 1,#pairRef do
        if hms[pairRef[i][1]]:max() > actThresh and hms[pairRef[i][2]]:max() > actThresh then
            -- Set appropriate line color
            local color
            if partColor[pairRef[i][1]] == 1 then color = {0,.3,1}
            elseif partColor[pairRef[i][1]] == 2 then color = {1,.3,0}
            elseif partColor[pairRef[i][1]] == 3 then color = {0,0,1}
            elseif partColor[pairRef[i][1]] == 4 then color = {1,0,0}
            else color = {.7,0,.7} end

            -- Draw line
            im = drawLine(im, coords[pairRef[i][1]], coords[pairRef[i][2]], 4, color, 0)
        end
    end

    return im
end

function drawSkeleton(input, hms, coords)

    local im = input:clone()

    local pairRef = {
        {1,2},      {2,3},      {3,7},
        {4,5},      {4,7},      {5,6},
        {7,9},      {9,10},
        {14,9},     {11,12},    {12,13},
        {13,9},     {14,15},    {15,16}
    }

    local partNames = {'RAnk','RKne','RHip','LHip','LKne','LAnk',
                       'Pelv','Thrx','Neck','Head',
                       'RWri','RElb','RSho','LSho','LElb','LWri'}
    local partColor = {1,1,1,2,2,2,0,0,0,0,3,3,3,4,4,4}

    local actThresh = 0.2

    -- Loop through adjacent joint pairings
    for i = 1,#pairRef do
        if hms[pairRef[i][1]]:max() > actThresh and hms[pairRef[i][2]]:max() > actThresh then
            -- Set appropriate line color
            local color
            if partColor[pairRef[i][1]] == 1 then color = {0,.3,1}
            elseif partColor[pairRef[i][1]] == 2 then color = {1,.3,0}
            elseif partColor[pairRef[i][1]] == 3 then color = {0,0,1}
            elseif partColor[pairRef[i][1]] == 4 then color = {1,0,0}
            else color = {.7,0,.7} end

            -- Draw line
            im = drawLine(im, coords[pairRef[i][1]], coords[pairRef[i][2]], 4, color, 0)
        end
    end

    return im
end

function isHuman(hms)

    local pairRef = {
        {1,2},      {2,3},      {3,7},
        {4,5},      {4,7},      {5,6},
        {7,9},      {9,10},
        {14,9},     {11,12},    {12,13},
        {13,9},     {14,15},    {15,16}
    }

    local actThresh = 0.01

    local countParts = 0

    -- Loop through adjacent joint pairings
    for i = 1,#pairRef do
        if hms[pairRef[i][1]]:mean() > actThresh and hms[pairRef[i][2]]:mean() > actThresh then
            -- Count
            countParts = countParts + 1
        end
    end
    if countParts<10 then
        return false
    else
        return true
    end

end


function drawOutput(input, hms, coords)
    local im = drawSkeleton(input, hms, coords)

    local colorHms = {}
    local inp64 = image.scale(input,64):mul(.3)
    for i = 1,16 do 
        colorHms[i] = colorHM(hms[i])
        colorHms[i]:mul(.7):add(inp64)
    end
    local totalHm = compileImages(colorHms, 4, 4, 64)
    im = compileImages({im,totalHm}, 1, 2, 256)
    im = image.scale(im,756)
    return im
end

function drawOutputAic(input, hms, coords,inpres, outres)
    if inpres == nil then inpres = 256 end
    if outres == nil then outres = 64 end
    local im = drawSkeletonAic(input, hms, coords)
    if hms:size():size() == 4 then hms = hms:view(hms:size(2), hms:size(3), hms:size(4)) end
    
    local colorHms = {}
    local inp64 = image.scale(input,outres):mul(.3)
    for i = 1,14 do
        colorHms[i] = colorHM(hms[i])
        colorHms[i]:mul(.7):add(inp64)
    end
    local totalHm = compileImages(colorHms, 4, 4, outres)
    im = compileImages({im,totalHm}, 1, 2, inpres)
    im = image.scale(im,756)
    return im
end

function drawOutputcoco(input, hms, coords,inpresH, inpresW, outresH, outresW)
    --if inpres == nil then inpres = 256 end
    --if outres == nil then outres = 64 end
    local im = drawSkeletoncoco(input, hms, coords)
    if hms:size():size() == 4 then hms = hms:view(hms:size(2), hms:size(3), hms:size(4)) end
    local colorHms = {}
    local inp64 = image.scale(input,outresW,outresH):mul(.3)
    for i = 1,16 do
        colorHms[i] = colorHM(hms[i])
        colorHms[i]:mul(.7):add(inp64)
    end
    local totalHm = compileImages(colorHms, 4, 4, outresH, outresW)
    im = compileImages({im,totalHm}, 1, 2, inpresH, inpresW)
    im = image.scale(im,756)
    return im
end

function drawOutputcocoOrig(input, hms, coords)
    local img = input:clone()
    local im = drawSkeletoncoco(img, hms, coords)
    if hms:size():size() == 4 then hms = hms:view(hms:size(2), hms:size(3), hms:size(4)) end
    local colorHms = {}
    local inp64 = img:mul(.3)
    for i = 1,16 do
        colorHms[i] = colorHM(hms[i])
        inp64:add(colorHms[i]:mul(.7))
    end
    return im,inp64
end
-------------------------------------------------------------------------------
-- Functions for evaluation
-------------------------------------------------------------------------------

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

function distAccuracy(dists, thr)
    -- Return percentage below threshold while ignoring values with a -1
    if not thr then thr = .5 end
    if torch.ne(dists,-1):sum() > 0 then
        return dists:le(thr):eq(dists:ne(-1)):sum() / dists:ne(-1):sum()
    else
        return -1
    end
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

function fineTuneHeatmap(location, offsetX, offsetY)
    if type(location) == 'table' then
        return fineTuneHeatmap(location[opt.nStack], offsetX[opt.nStack],offsetY[opt.nStack])
    else
        if ##location ~= 2 then
            local ftHm = torch.zeros(#location)
            for i=1,(#location)[1] do ftHm[i] = fineTuneHeatmap(location[i],offsetX[i],offsetY[i]) end
            return ftHm
        else
            local ftHm = torch.zeros(location:size())
            local tmpHm = torch.zeros(location:size())
            local loc = location:clone()
            local offx = offsetX:clone()
            local offy = offsetY:clone()
            local ht = (#location)[1]
            local wd = (#location)[2]
            for i=1,ht do tmpHm[i]:indexAdd(1,torch.clamp(torch.range(1,wd):long()+offx[i]:round():long(),1,loc:size(2)),loc[i]:double()) end
            for i=1,wd do ftHm[{{},i}]:indexAdd(1,torch.clamp(torch.range(1,ht):long()+offy[{{},i}]:round():long(),1,loc:size(1)),tmpHm[{{},i}]:double()) end
            return ftHm
        end
    end
end


function loadInp(argv, a, idxs, i)
    local im = image.load(argv[2] .. a['images'][idxs[i]],3)
    --sub mean
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
    local ht = a['ymax'][idxs[i]]-a['ymin'][idxs[i]]
    local width = a['xmax'][idxs[i]]-a['xmin'][idxs[i]]
    local scaleRate
    if width > 100 then
        scaleRate = 0.2
    else
        scaleRate = 0.3
    end
    pt1[1] = math.max(1,(pt1[1] - width*scaleRate/2))
    pt1[2] = math.max(1,(pt1[2] - ht*scaleRate/2))
    pt2[1] = math.max(math.min(imgwidth+1,(pt2[1] + width*scaleRate/2)),pt1[1]+5)
    pt2[2] = math.max(math.min(imght+1,(pt2[2] + ht*scaleRate/2)),pt1[2]+5)
    local inputResH = 320
    local inputResW = 256
    local outResH = 80
    local outResW = 64
    --local inp = crop(im, center, scale, 0, inputRes)
    local inp = cropBox(im, pt1:int(), pt2:int(), 0, inputResH, inputResW)
    return inp, pt1, pt2
end
