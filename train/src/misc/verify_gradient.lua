-- This hasn't been used in a while, but does a gradient check for custom layer modules

paths.dofile('../net/setup.lua')   -- Initialize options and load data
paths.dofile('../net/model.lua')   -- Read in network model

local idx = 1
local batch_size = 2
local inp_dims = {{idx,idx+batch_size-1}}
for j = 1,dataDim:size()[1] do inp_dims[j+1] = {1,dataDim[j]} end
local label_dims = {{idx,idx+batch_size-1}}
for j = 1,labelDim:size()[1] do label_dims[j+1] = {1,labelDim[j]} end
local inputs = trainFile:read('data'):partial(unpack(inp_dims))
local labels = trainFile:read('label'):partial(unpack(label_dims))
in_size = inputs:size()

local function check_net(input)
    input = input:view(in_size)
    if opt.GPU ~= -1 then
        input = torch.CudaTensor(input:size()):copy(input)
        labels = torch.CudaTensor(labels:size()):copy(labels)
    end
    if preprocess then input,labels = preprocess(input,labels) end
    local output = model:forward(input)
    local loss = criterion:forward(output,labels)
    local d_loss = criterion:backward(output,labels)
    local grad = model:backward(input, d_loss)
    if unprocess then grad = unprocess(grad, batch_size) end
    grad = grad:view(grad:numel())
    return loss, grad
end

timer = torch.Timer()
diff,a,b = optim.checkgrad(check_net, inputs:view(inputs:numel()), 1e-7)
print(a:cat(b,2))
print(diff)
print("Time: " .. timer:time().real)

trainFile:close()
