require 'paths'
arg = {'-GPU','-1'}
paths.dofile('../ref.lua')
require 'sys'

pairRef = {
    {1,2},      {2,3},      {3,7},
    {4,5},      {4,7},      {5,6},
    {7,9},      {9,10},
    {14,9},     {11,12},    {12,13},
    {13,9},    {14,15},    {15,16}
}

partNames = {'RLAnk','RLKne','RLHip','LLHip','LLKne','LLAnk',
			 'Pelv','Thrx','Neck','Head',
			 'RUWri','RUElb','RUSho','LUSho','LUElb','LUWri'}

function transformCoords(i,cds)
	local c = annot['test']['center'][i]
	local s = annot['test']['scale'][i]
	local new_cds = torch.zeros(cds:size())
	for j = 1,cds:size(1) do
		new_cds[j] = transform(cds[j]:add(-.5),c,s,0,64,true)
	end
	return new_cds
end

predsfile = opt.expDir .. '/best/preds_full.h5'
preds = hdf5.open(predsfile)

function getImg(idx,res)
	hms = preds:read('pred_heatmaps'):partial({idx,idx},{1,16},{1,64},{1,64})
	coord = preds:read('preds_tf'):partial({idx,idx},{1,16},{1,2})[1]
	im = image.load(opt.dataDir .. '/images/' .. annot['test']['images'][idx])
	local c = annot['test']['center'][idx]
	local s = annot['test']['scale'][idx]
	for j = 1,#pairRef do
		if hms[1][pairRef[j][1]]:max() > .1 and hms[1][pairRef[j][2]]:max() > .1 then
			if string.sub(partNames[pairRef[j][1]],1,1) == 'L' then
				drawLine(im[1],coord[pairRef[j][1]],coord[pairRef[j][2]],4*s,1,0,2)
				drawLine(im[2],coord[pairRef[j][1]],coord[pairRef[j][2]],4*s,1,0,2)
				drawLine(im[3],coord[pairRef[j][1]],coord[pairRef[j][2]],4*s,1,1,2)
			elseif string.sub(partNames[pairRef[j][1]],1,1) == 'R' then
				drawLine(im[1],coord[pairRef[j][1]],coord[pairRef[j][2]],4*s,1,1,2)
				drawLine(im[2],coord[pairRef[j][1]],coord[pairRef[j][2]],4*s,1,0,2)
				drawLine(im[3],coord[pairRef[j][1]],coord[pairRef[j][2]],4*s,1,0,2)
			else
				drawLine(im[1],coord[pairRef[j][1]],coord[pairRef[j][2]],4*s,1,.7,2)
				drawLine(im[2],coord[pairRef[j][1]],coord[pairRef[j][2]],4*s,1,0,2)
				drawLine(im[3],coord[pairRef[j][1]],coord[pairRef[j][2]],4*s,1,.7,2)
			end
		end
	end
	im = crop(im, c, s, 0, res)
	return im
end

function compileImages(imgs, nrows, ncols, res)
	print(imgs[1]:size())
	local totalImg = torch.zeros(3,nrows*res,ncols*res)
	for i = 1,#imgs do
		local r = torch.floor((i-1)/ncols) + 1
		local c = ((i - 1) % ncols) + 1
		print(r)
		print(c)
		totalImg:sub(1,3,(r-1)*res+1,r*res,(c-1)*res+1,c*res):copy(imgs[i])
	end
	w = image.display{image=totalImg,win=w}
	return totalImg
end

num_imgs = 10
test_idxs = torch.randperm(11000):sub(1,12)
num_imgs = test_idxs:numel()
ims = {}
for i = 1,num_imgs do
	print(test_idxs[i])
	ims[i] = getImg(test_idxs[i],728)
end
final = compileImages(ims, 6, 2, 728)
image.savePNG('examples.png',final)
