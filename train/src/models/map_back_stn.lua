require 'stn'

function createMapBackStn(inpHeight,inpWidth)
   local trans = nn.Transpose({2,3},{3,4})
   local grids = nn.AffineGridGeneratorBHWD(inpHeight,inpWidth)
   local locNet = nn.ParallelTable():add(trans):add(grids)
   local Sampling = nn.BilinearSamplerBHWD()
   local output = nn.Transpose({3,4},{2,3})
   local model = nn.Sequential():add(locNet):add(Sampling):add(output)
   return model
end

