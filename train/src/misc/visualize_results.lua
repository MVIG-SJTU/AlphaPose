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

partNames = {'RAnk','RKne','RHip','LHip','LKne','LAnk',
			 'Pelv','Thrx','Neck','Head',
			 'RWri','RElb','RSho','LSho','LElb','LWri'}

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

for i = 1,10000 do
	hms = preds:read('pred_heatmaps'):partial({i,i},{1,16},{1,64},{1,64})
	print(i)
	old_coords = preds:read('preds'):partial({i,i},{1,16},{1,2})
	new_coord = transformCoords(i,old_coords[1])
	new_coords_2 = preds:read('preds_tf'):partial({i,i},{1,16},{1,2})
	im = image.load(opt.dataDir .. '/images/' .. annot['test']['images'][i])
	for j = 1,#pairRef do
		if hms[1][pairRef[j][1]]:max() > .05 and hms[1][pairRef[j][2]]:max() > .05 then
			local s = annot['test']['scale'][i]
			if string.sub(partNames[pairRef[j][1]],1,1) == 'L' then
				drawLine(im[1],new_coord[pairRef[j][1]],new_coord[pairRef[j][2]],4*s,1,0,2)
				drawLine(im[2],new_coord[pairRef[j][1]],new_coord[pairRef[j][2]],4*s,1,0,2)
				drawLine(im[3],new_coord[pairRef[j][1]],new_coord[pairRef[j][2]],4*s,1,1,2)
			elseif string.sub(partNames[pairRef[j][1]],1,1) == 'R' then
				drawLine(im[1],new_coord[pairRef[j][1]],new_coord[pairRef[j][2]],4*s,1,1,2)
				drawLine(im[2],new_coord[pairRef[j][1]],new_coord[pairRef[j][2]],4*s,1,0,2)
				drawLine(im[3],new_coord[pairRef[j][1]],new_coord[pairRef[j][2]],4*s,1,0,2)
			else
				drawLine(im[1],new_coord[pairRef[j][1]],new_coord[pairRef[j][2]],4*s,1,.7,2)
				drawLine(im[2],new_coord[pairRef[j][1]],new_coord[pairRef[j][2]],4*s,1,0,2)
				drawLine(im[3],new_coord[pairRef[j][1]],new_coord[pairRef[j][2]],4*s,1,.7,2)
			end
		else
			print("Not drawing:",partNames[pairRef[j][1]],partNames[pairRef[j][2]])
		end
	end
	w = image.display{image=im,win=w}
	w2 = image.display{image=hms[1],win=w2}
	sys.sleep(.2)
end
