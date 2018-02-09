
load('../../data/mpii/annot/mpii_human_pose_v1_u12_1.mat');
imgname = h5read('../../data/mpii/annot/test.h5', '/imgname');
index = h5read('../../data/mpii/annot/test.h5', '/index') + 1;
person = h5read('../../data/mpii/annot/test.h5', '/person') + 1;
scale = h5read('../../data/mpii/annot/test.h5', '/scale');
preds = h5read('../../exp/mpii/best/final_preds.h5', '/preds');
nJoints = size(preds, 2);

for i = 1:size(preds, 3)
    idx = index(i);
    assert(strcmp(RELEASE.annolist(idx).image.name, imgname{i}) == 1);
    assert(RELEASE.img_train(idx) == 0);

    x = cell(1, nJoints);
    y = cell(1, nJoints);
    id = cell(1, nJoints);
    for j = 1:nJoints
        x{j} = preds(1, j, i);
        y{j} = preds(2, j, i);
        id{j} = j - 1;
    end
    RELEASE.annolist(idx).annorect(person(i)).annopoints = struct('point', struct('x', x, 'y', y,'id', id));
end

save('mpii-prediction-01.mat', 'RELEASE');
