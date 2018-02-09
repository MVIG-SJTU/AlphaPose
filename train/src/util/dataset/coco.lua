local M = {}
Dataset = torch.class('pose.Dataset',M)

function Dataset:__init()
    self.nJoints_coco = 17
    self.nJoints_mpii = 16
    self.nJoints = 33

    self.accIdxs_coco = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17}
    self.accIdxs_mpii = {18,19,20,21,22,23,28,29,32,33}
    self.accIdxs = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,
                    18,19,20,21,22,23,28,29,32,33}

    self.flipRef_coco = {{2,3},   {4,5},   {6,7},
                        {8,9}, {10,11}, {12,13},
                        {14,15}, {16,17}}
    self.flipRef_mpii = {{1,6},   {2,5},   {3,4},
                        {11,16}, {12,15}, {13,14}}
    self.flipRef = {{2,3},   {4,5},   {6,7},
                    {8,9}, {10,11}, {12,13},
                    {14,15}, {16,17},   -- coco
                    {18,23}, {19,22}, {20,21},
                    {28,33}, {29,32}, {30,31}} -- mpii

    self.skeletonRef = {{1,6,1},      {1,7,1},
                            {6,8,2},    {7,9,2},    {8,10,2},      {9,11,2},
                            {1,12,3},      {1,13,3},
                            {12,14,4},     {14,16,4},   {13,15,4},     {15,17,4}}


    local annot = {}
    local tags = {'imgname','part','bndbox'}
    local a_coco = hdf5.open('../data/coco/annot.h5','r')
    local a_mpii = hdf5.open('../data/mpii/annot.h5','r')

    for _,tag in ipairs(tags) do annot[tag..'_coco'] = a_coco:read(tag):all() end
    for _,tag in ipairs(tags) do annot[tag..'_mpii'] = a_mpii:read(tag):all() end
    a_coco:close()
    a_mpii:close()
    annot.part_coco:add(1)
    annot.part_mpii:add(1)

    -- Index reference
    if not opt.idxRef then
        opt.idxRef = {}
        -- Set up training/validation split
        opt.idxRef.train = torch.cat(torch.range(1,annot.part_coco:size(1)-5887),
                                    torch.range(annot.part_coco:size(1)+1,annot.part_coco:size(1)+annot.part_mpii:size(1)-1358))
        opt.idxRef.valid = torch.range(annot.part_coco:size(1)-5887,annot.part_coco:size(1))

        torch.save(opt.save .. '/options.t7', opt)
    end

    self.size_coco = annot.part_coco:size(1)
    self.size_mpii = annot.part_mpii:size(1)

    self.annot = annot
    self.nsamples = {train=opt.idxRef.train:numel(),
                     valid=opt.idxRef.valid:numel()}
end

function Dataset:size(set)
    return self.nsamples[set]
end

function Dataset:getPath(idx)
    if idx <= self.size_coco then
        return paths.concat(opt.dataDir,'coco','images',ffi.string(self.annot.imgname_coco[idx]:char():data()))
    else
        return paths.concat(opt.dataDir,'mpii','images',ffi.string(self.annot.imgname_mpii[idx-self.size_coco]:char():data()))
    end
end

function Dataset:loadImage(idx)
    return image.load(self:getPath(idx),3)
end

function Dataset:getPartInfo(idx)
    local pts, bndbox, imgset
    if idx <= self.size_coco then
        pts = self.annot.part_coco[idx]:clone()
        bndbox = self.annot.bndbox_coco[idx]:clone()
        imgset = 'coco'
    else
        pts = self.annot.part_mpii[idx-self.size_coco]:clone()
        bndbox = self.annot.bndbox_mpii[idx-self.size_coco]:clone()
        imgset = 'mpii'
    end
    return pts, bndbox, imgset
end


return M.Dataset
