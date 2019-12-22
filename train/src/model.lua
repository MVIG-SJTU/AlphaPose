require 'cutorch'
require 'cunn'
require 'cudnn'
--- Load up network model or initialize from scratch
paths.dofile('models/' .. opt.netType .. '.lua')

-- Continuing an experiment where it left off
if opt.continue or opt.branch ~= 'none' then
    local prevModel = opt.load .. '/final_model.t7'
    print('==> Loading model from: ' .. prevModel)
    model = torch.load(prevModel)
    if opt.nGPU > 1 then
        model = makeDPT(model, opt.nGPU)
    end
-- For loading model with parallel sppe
elseif opt.addParallelSPPE ==  true and opt.addSSTN == false then
    assert(paths.filep(opt.loadModel), 'File not found: ' .. opt.loadModel)
    print('==> Loading model from: ' .. opt.loadModel)
    model = torch.load(opt.loadModel)
    --print (model.forwardnodes[26].data.module:listModules())
    model_paraSPPE = model.forwardnodes[26].data.module
    for i, m in ipairs(model_paraSPPE.modules) do
        m.accGradParameters = function() end
        m.updateParameters = function() end
    end
-- -- Add parallel sppe
-- elseif opt.addParallelSPPE == true then
--     assert(paths.filep(opt.loadModel), 'File not found: ' .. opt.loadModel)
--     print('==> Loading model from: ' .. opt.loadModel..' and add parallel SPPE')
--     local model_sppe = torch.load(opt.loadModel)
--     local model_paraSPPE = torch.load(opt.loadModel)
--     local model_sdtn = {}
--     for i = 1,opt.nStack do
--       local tmp_sdtn = createMapBackStn(opt.outputRes,opt.outputRes)
--       table.insert(model_sdtn,tmp_sdtn)
--     end
--     local inp = nn.Identity()()
--     local img_stn,_theta = spanet(inp):split(2)
--     local out_sppe_stn = {model_sppe(img_stn):split(opt.nStack)}
--     local out_paraSPPE_stn = {model_paraSPPE(img_stn):split(opt.nStack)}
--     local alpha = nn.Get_Alpha()(_theta)
--     local out = {}
--     for i = 1,opt.nStack do
--       local tmp_out = model_sdtn[i]({out_sppe_stn[i],alpha})
--       table.insert(out,tmp_out)
--     end
--     for i = 1,opt.nStack do
--       table.insert(out,out_paraSPPE_stn[i])
--     end
--     model = nn.gModule({inp}, out)

-- Add parallel sppe
elseif opt.addParallelSPPE == true then
    assert(paths.filep(opt.loadModel), 'File not found: ' .. opt.loadModel)
    print('==> Loading model from: ' .. opt.loadModel..' and add parallel SPPE')
    local model_sppe = torch.load(opt.loadModel)
    print('==> Loading parallel SPPE from: COCO_4stacked.t7')
    local model_paraSPPE = torch.load('COCO_4stacked.t7')
    print('==> Freezing layers of parallel SPPE')
    for i, m in ipairs(model_paraSPPE.modules) do
      if torch.type(m):find('Convolution') then
        m.accGradParameters = function() end
        m.updateParameters = function() end
      end
    end
    local model_sdtn = {}
    for i = 1,opt.nStack do
      local tmp_sdtn = createMapBackStn(opt.outputRes,opt.outputRes)
      table.insert(model_sdtn,tmp_sdtn)
    end
    local inp = nn.Identity()()
    local img_stn,_theta = spanet(inp):split(2)
    local out_sppe_stn = {model_sppe(img_stn):split(opt.nStack)}
    local out_paraSPPE_stn = {model_paraSPPE(img_stn):split(opt.nParaStack)}
    local alpha = nn.Get_Alpha()(_theta)
    local out = {}
    for i = 1,opt.nStack do
      local tmp_out = model_sdtn[i]({out_sppe_stn[i],alpha})
      table.insert(out,tmp_out)
    end
    for i = 1,opt.nParaStack do
      table.insert(out,out_paraSPPE_stn[i])
    end
    model = nn.gModule({inp}, out)

-- Or we add a STN to a trained model
elseif opt.addSSTN ~= false and opt.loadModel ~= 'none' then
    assert(paths.filep(opt.loadModel), 'File not found: ' .. opt.loadModel)
    print('==> Loading model from: ' .. opt.loadModel)
    local model_base = torch.load(opt.loadModel)
    local model_sdtn = {}
    for i = 1,opt.nStack do
      local tmp_sdtn = createMapBackStn(opt.outputRes,opt.outputRes)
      table.insert(model_sdtn,tmp_sdtn)
    end
    local inp = nn.Identity()()
    local img_stn,_theta = spanet(inp):split(2)
    local out_base = {model_base(img_stn):split(opt.nStack)}
    local alpha = nn.Get_Alpha()(_theta)
    local out = {}
    for i = 1,opt.nStack do
      local tmp_out = model_sdtn[i]({out_base[i],alpha})
      table.insert(out,tmp_out)
    end
    model = nn.gModule({inp}, out)
    --   graph.dot(model.fg, 'Forward Graph')


-- Or we add a STN to a new model
elseif opt.addSSTN ~= false and opt.loadModel == 'none' then
    print('==> Creating model from file: models/' .. opt.netType .. '.lua')
    local model_base = createModel(modelArgs)
    local model_sdtn = {}
    for i = 1,opt.nStack do
      local tmp_sdtn = createMapBackStn(opt.outputRes,opt.outputRes)
      table.insert(model_sdtn,tmp_sdtn)
    end
    local inp = nn.Identity()()
    local img_stn,theta = spanet(inp):split(2)
    local out_base = {model_base(img_stn):split(opt.nStack)}
    local alpha = nn.Get_Alpha()(theta)
    local out = {}
    for i = 1,opt.nStack do
      local tmp_out = model_sdtn[i]({out_base[i],alpha})
      table.insert(out,tmp_out)
    end
    model = nn.gModule({inp}, out)
--    graph.dot(model.fg,'Model')

-- Or a path to previously trained model is provided
elseif opt.loadModel ~= 'none' then
    assert(paths.filep(opt.loadModel), 'File not found: ' .. opt.loadModel)
    print('==> Loading model from: ' .. opt.loadModel)
    model = torch.load(opt.loadModel)
    if opt.nGPU > 1 then
        model = makeDPT(model, opt.nGPU)
    end
    --graph.dot(model.fg, 'Forward Graph', opt.save..'vismodel')

-- Or we're starting fresh
else
    print('==> Creating model from file: models/' .. opt.netType .. '.lua')
    model = createModel(modelArgs)
    if opt.nGPU > 1 then
        model = makeDPT(model, opt.nGPU)
    end
    --graph.dot(model.fg, 'Forward Graph', opt.save..'vismodel')
end

-- Criterion (can be set in the opt.task file as well)
if not criterion then
    criterion = nn[opt.crit .. 'Criterion']()
end

if opt.GPU ~= -1 then
    -- Convert model to CUDA
    print('==> Converting model to CUDA')
    model:cuda()
    criterion:cuda()

    cudnn.fastest = true
    cudnn.benchmark = true
else
    print('==> Converting model to CPU')
    model = cudnn.convert(model,nn):float()
    criterion = criterion:float()
end
