-- Residual Pyramid with Pooling
local nnlib = require('cudnn')
local conv = nnlib.SpatialConvolution
local batchnorm = nnlib.SpatialBatchNormalization
local relu = nnlib.ReLU

-- Main convolutional block
local function convBlock(numIn,numOut,inputResH,inputResW,type,baseWidth, cardinality,stride)
    local model = nn.Sequential()
    local addTable = nn.ConcatTable()

    -- main branch
    local s = nn.Sequential()
    if type ~= 'no_preact' then
        s:add(batchnorm(numIn))
        s:add(relu(true))        
    end

    s:add(conv(numIn,numOut/2,1,1))
    s:add(batchnorm(numOut/2))
    s:add(relu(true))
    s:add(conv(numOut/2,numOut/2,3,3,stride,stride,1,1))
    addTable:add(s)

    -- Pyramid
    local function pyramid(D, C)
        local pyraTable = nn.ConcatTable()
        local sc = 2 ^(1/C);
        for i = 1, C do
            local scaled = 1/(sc^i)
            local s = nn.Sequential()
                :add(nn.SpatialFractionalMaxPooling(2, 2, scaled, scaled))
                :add(conv(D,D,3,3,1,1,1,1))
                :add(nn.SpatialUpSamplingBilinear({oheight=inputResH, owidth=inputResW}))
            pyraTable:add(s)
        end
        local pyra = nn.Sequential()
            :add(pyraTable)
            :add(nn.CAddTable(false))
        return pyra
    end


    local D = math.floor(numOut/baseWidth)
    local C = cardinality
    local s = nn.Sequential()

    if type ~= 'no_preact' then
        s:add(batchnorm(numIn))
        s:add(relu(true))        
    end
    local a = conv(D,numOut/2,1,1)
    a.nBranchIn = C
    s:add(conv(numIn,D,1,1,stride,stride))
    s:add(batchnorm(D))
    s:add(relu(true))
    s:add(pyramid(D, C))
    s:add(batchnorm(D))
    s:add(relu(true))
    -- s:add(conv(D,numOut/2,1,1))
    s:add(a)

    -- add together
    addTable:add(s)
    
    local elewiseAdd = nn.Sequential()
        :add(addTable)
        :add(nn.CAddTable(false))

    -- combine model
    return model:add(elewiseAdd)
        :add(batchnorm(numOut/2))
        :add(relu(true))
        :add(conv(numOut/2,numOut,1,1))
end

-- Skip layer
local function skipLayer(numIn,numOut,stride, useConv)
    if useConv then
        return nn.Sequential()
            :add(batchnorm(numIn))
            :add(relu(true))        
            :add(conv(numIn,numOut,1,1,stride,stride))
    end

    if numIn == numOut and stride == 1 then
        return nn.Identity()
    else
        return nn.Sequential()
            :add(batchnorm(numIn))
            :add(relu(true))        
            :add(conv(numIn,numOut,1,1,stride,stride))
    end
end

-- Residual block
--function Residual(numIn,numOut,stride,type,useConv,inputRes,baseWidth,cardinality)
--    local type = type or 'preact'
--    local baseWidth = baseWidth or 9
--    local cardinality = cardinality or 4
--    local useConv = useConv or false
--    local stride = stride or 1
--    return nn.Sequential()
--        :add(nn.ConcatTable()
--            :add(convBlock(numIn,numOut,inputRes,type,baseWidth,cardinality,stride))
--            :add(skipLayer(numIn,numOut,stride,useConv)))
--        :add(nn.CAddTable(true))
--end
function Residual(numIn,numOut,stride,type,useConv,inputResH,inputResW,baseWidth,cardinality)
    local type = type or 'preact'
    local baseWidth = baseWidth or 9
    local cardinality = cardinality or 4
    local useConv = useConv or false
    local stride = stride or 1
    return nn.Sequential()
        :add(nn.ConcatTable()
            :add(convBlock(numIn,numOut,inputResH,inputResW,type,baseWidth,cardinality,stride))
            :add(skipLayer(numIn,numOut,stride,useConv)))
        :add(nn.CAddTable(true))
end

return Residual
