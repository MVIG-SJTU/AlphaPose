-- Pyramid Residual Module
local nn = require 'nn'
local cunn = require 'cunn'
local cudnn = require 'cudnn'
local nngraph = require 'nngraph'
local Residual = require('models.layers.Residual')
local ResidualPyramid = require('models.layers.PRM')
local nnlib = cudnn

local function hourglass(n, f, nModules, inp, inputRes, type, B, C)
    local ResidualUp = n >= 2 and ResidualPyramid or Residual
    local ResidualDown = n >= 3 and ResidualPyramid or Residual

    -- Upper branch
    local up1 = inp
    for i = 1,nModules do up1 = ResidualUp(f,f,1,type,false,inputRes,B,C)(up1) end

    -- Lower branch
    local low1 = nnlib.SpatialMaxPooling(2,2,2,2)(inp)
    for i = 1,nModules do low1 = ResidualDown(f,f,1,type,false, inputRes/2,B,C)(low1) end
    local low2

    if n > 1 then low2 = hourglass(n-1,f,nModules,low1,inputRes/2,type,B,C)
    else
        low2 = low1
        for i = 1,nModules do low2 = ResidualDown(f,f,1,type,false,inputRes/2,B,C)(low2) end
    end

    local low3 = low2
    for i = 1,nModules do low3 = ResidualDown(f,f,1,type,true,inputRes/2,B,C)(low3) end
    local up2 = nn.SpatialUpSamplingNearest(2)(low3)

    -- Bring two branches together
    return nn.CAddTable()({up1,up2})
end

local function lin(numIn,numOut,inp)
    -- Apply 1x1 convolution, stride 1, no padding
    local l = nnlib.SpatialConvolution(numIn,numOut,1,1,1,1,0,0)(inp)
    return nnlib.ReLU(true)(nnlib.SpatialBatchNormalization(numOut)(l))
end


local function preact(num, inp)    
    return nnlib.ReLU(true)(nnlib.SpatialBatchNormalization(num)(inp))
end

function createModel()
    local B, C = opt.baseWidth, opt.cardinality
    local inputRes = opt.inputRes/4

    local inp = nn.Identity()()

    -- Initial processing of the image
    local cnv1_ = nnlib.SpatialConvolution(3,64,7,7,2,2,3,3)(inp)           -- 128
    local cnv1 = nnlib.ReLU(true)(nnlib.SpatialBatchNormalization(64)(cnv1_))
    local r1 = ResidualPyramid(64,128,1,'no_preact',false,opt.inputRes/2, B,C)(cnv1)
    local pool = nnlib.SpatialMaxPooling(2,2,2,2)(r1)                       -- 64
    local r4 = ResidualPyramid(128,128,1,'preact',false,inputRes,B,C)(pool)
    local r5 = ResidualPyramid(128,opt.nFeats,1,'preact',false,inputRes,B,C)(r4)

    local out = {}
    local inter = r5

    for i = 1,opt.nStack do
        local hg = hourglass(4,opt.nFeats, opt.nResidual, inter, inputRes,'preact',B,C)

        -- Linear layer to produce first set of predictions
        local ll = preact(opt.nFeats, hg)
        ll = lin(opt.nFeats,opt.nFeats,ll)

        -- Predicted heatmaps
        local tmpOut = nnlib.SpatialConvolution(opt.nFeats,opt.nClasses,1,1,1,1,0,0)(ll)
        table.insert(out,tmpOut)

        -- Add predictions back
        if i < opt.nStack then
            local ll_ = nnlib.SpatialConvolution(opt.nFeats,opt.nFeats,1,1,1,1,0,0)(ll)
            local tmpOut_ = nnlib.SpatialConvolution(opt.nClasses,opt.nFeats,1,1,1,1,0,0)(tmpOut)
            inter = nn.CAddTable()({inter, ll_, tmpOut_})
        end
    end

    -- Final model
    local model = nn.gModule({inp}, out)

    return model:cuda()

end

return createModel