local GetAlpha, parent = torch.class('nn.Get_Alpha', 'nn.Module')

function GetAlpha:__init()
   parent.__init(self)
end

local function addOuterDim(t)
   local sizes = t:size()
   local newsizes = torch.LongStorage(sizes:size()+1)
   newsizes[1]=1
   for i=1,sizes:size() do
      newsizes[i+1]=sizes[i]
   end
   return t:view(newsizes)
end

function GetAlpha:updateOutput(input)
   local theta = input
   if (theta:nDimension() == 2) then
      theta = addOuterDim(theta)
   end
   assert ((theta:nDimension() == 3) and (theta:size(2) == 2) and (theta:size(3) == 3), 'Please make sure theta is 2x3 matrix')
   local alpha = torch.CudaTensor(theta:size()):zero()
   local batchsize = alpha:size(1)
   for i=1,batchsize do
      local theta_first_two_colomn = theta:select(1,i)[{{},{1,2}}]
      alpha:select(1,i)[{{},{1,2}}]:copy(torch.inverse(theta_first_two_colomn))
      alpha:select(1,i)[{{},3}] = torch.mv(alpha:select(1,i)[{{},{1,2}}],torch.mul(theta:select(1,i):select(2,3), -1))
   end
   self.output = alpha
   return alpha
end

function GetAlpha:updateGradInput(_input,gradOutput)
   if (_input:nDimension() == 2) then
      _input = addOuterDim(_input)
   end
   self.gradInput = torch.CudaTensor(_input:size()):fill(0)
   local batchsize = _input:size(1)

   for i = 1,batchsize do
     local theta = _input:select(1,i)
     local alpha = self.output:select(1,i)
     local dy = gradOutput:select(1,i)
     local dtheta3 = (-1)*torch.mv(alpha[{{},{1,2}}]:t(), dy[{{},3}])
     local dalpha1_2 = torch.CudaTensor(2,2):zero()
     dalpha1_2:addr(dy[{{},3}], theta[{{},3}]):mul(-1):add(dy[{{},{1,2}}])
     dalpha1_2 = dalpha1_2:reshape(4)

     local theta1_2 = theta[{{},{1,2}}]:reshape(4)
     local det = theta1_2[1]*theta1_2[4]-theta1_2[2]*theta1_2[3]
     local da_dt = torch.zeros(4,4)
     -- assign the value of each entry to matrix da_dt
     da_dt[{1,1}] = (-1)*theta1_2[4]*theta1_2[4];   da_dt[{1,2}] = theta1_2[3]*theta1_2[4];   da_dt[{1,3}] = theta1_2[2]*theta1_2[4];  da_dt[{1,4}] = (-1)*theta1_2[2]*theta1_2[3];
     da_dt[{2,1}] = theta1_2[4]*theta1_2[2];  da_dt[{2,2}] = (-1)*theta1_2[1]*theta1_2[4];   da_dt[{2,3}] = (-1)*theta1_2[2]*theta1_2[2];  da_dt[{2,4}] = theta1_2[1]*theta1_2[2];
     da_dt[{3,1}] = theta1_2[4]*theta1_2[3];  da_dt[{3,2}] = (-1)*theta1_2[3]*theta1_2[3];  da_dt[{3,3}] = (-1)*theta1_2[1]*theta1_2[4]; da_dt[{3,4}] = theta1_2[1]*theta1_2[3]
     da_dt[{4,1}] = (-1)*theta1_2[2]*theta1_2[3];  da_dt[{4,2}] = theta1_2[1]*theta1_2[3];  da_dt[{4,3}] = theta1_2[2]*theta1_2[1];  da_dt[{4,4}] = (-1)*theta1_2[1]*theta1_2[1]
     --divided by the square of determinant
     da_dt:div(det*det)

     local dtheta1_2 = torch.CudaTensor(4)
     dtheta1_2 = torch.mv(da_dt:t():cuda(),dalpha1_2)
     dtheta1_2 = dtheta1_2:reshape(2,2)
     self.gradInput:select(1,i)[{{},{1,2}}]:copy(dtheta1_2)
     self.gradInput:select(1,i)[{{},3}]:copy(dtheta3)
  end
   return self.gradInput
end
