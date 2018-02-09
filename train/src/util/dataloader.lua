--
--  Original version: Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--  (Modified a bit by Alejandro Newell)
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Multi-threaded data loader
--

local Threads = require 'threads'
Threads.serialization('threads.sharedserialize')

local M = {}
local DataLoader = torch.class('DataLoader', M)

function DataLoader.create(opt, dataset, ref)
   -- The train and valid loader
   local loaders = {}

   for i, split in ipairs{'train', 'valid'} do
      if opt[split .. 'Iters'] > 0 then
         loaders[split] = M.DataLoader(opt, dataset, ref, split)
      end
   end

   return loaders
end

function DataLoader:__init(opt, dataset, ref, split)
    local function preinit()
        paths.dofile('dataset/' .. opt.dataset .. '.lua')
    end

    local function init()
        _G.opt, _G.dataset, _G.ref, _G.split = opt, dataset, ref, split
        paths.dofile('../ref.lua')
    end

    local function main(idx)
        torch.setnumthreads(1)
        return dataset:size(split)
    end

    local threads, sizes = Threads(opt.nThreads, preinit, init, main)
    self.threads = threads
    self.iters = opt[split .. 'Iters']
    self.batchsize = opt[split .. 'Batch']
    self.nsamples = sizes[1][1]
    self.split = split
end

function DataLoader:size()
    return self.iters
end

function DataLoader:run()
    local threads = self.threads
    local size = self.iters * self.batchsize

    local idxs = torch.range(1,self.nsamples)
    for i = 2,math.ceil(size/self.nsamples) do
        idxs = idxs:cat(torch.range(1,self.nsamples))
    end
    -- Shuffle indices
    idxs = idxs:index(1,torch.randperm(idxs:size(1)):long())
    -- Map indices to training/validation/test split
    idxs = opt.idxRef[self.split]:index(1,idxs:long())
    local n, idx, sample = 0, 1, nil
    local function enqueue()
        while idx <= size and threads:acceptsjob() do
            local indices = idxs:narrow(1, idx, math.min(self.batchsize, size - idx + 1))
            threads:addjob(
                function(indices)
                    local inp,out,mask = _G.loadData(_G.split, indices)
                    collectgarbage()
                    return {inp,out,mask,indices}
                end,
                function(_sample_) sample = _sample_ end, indices
            )
            idx = idx + self.batchsize
        end
    end

    local function loop()
        enqueue()
        if not threads:hasjob() then return nil end
        threads:dojob()
        if threads:haserror() then threads:synchronize() end
        enqueue()
        n = n + 1
        return n, sample
    end

    return loop
end

return M.DataLoader
