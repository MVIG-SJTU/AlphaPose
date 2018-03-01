function visualize_posetrack(video_set, labels_type, save_examples)
% The function is part of PoseTrack dataset. 
%
% It visualizes the annotated poses of all persons and their identitiy 
% labels. In PoseTrack each person is assigned with a unique identity label
% which will be represented by a unique color in this visualization. 
%
% video_set (string)  : video set to be visualized. 'train' or 'val'
% labels_type (string): 'all' or 'labeled_only'
%                       'all': all frames included unlabeled frames will be
%                              visualized
%                        'labeled_only': visualizes only labeled frames
% save_examples (bool): if '1', images with overlayed annotations will 
%                       be saved under './example/annotations'
% Usage:
%       visualize_posetrack(video_set, labels_type, save_examples)
%
% Example: 
%       visualize_posetrack('train', 'all', 0);
%       visualize_posetrack('val', 'labeled_only', 1);
%

if(nargin < 1)
    help visualize_posetrack
    return
end

if(nargin < 2)
    labels_type = 'labeled_only';
end

if(nargin < 3)
    save_examples = 0;
end

addpath('utils')

ROOT_DIR = '../';

ann_dir = fullfile(ROOT_DIR, '/annotations/', video_set);

fprintf(sprintf('Annotation directory: %s\n', ann_dir));
fprintf(sprintf('Visualizing %s set.\n', video_set));

annotations = dir(fullfile(ann_dir, '*.mat'));
num_videos  = length(annotations);

fprintf(sprintf('Total number of videos in %s set: %d\n', ...
                                        video_set, num_videos));

if(save_examples)
    example_ann_dir = fullfile(ROOT_DIR, 'annotation_examples', ... 
                                        video_set, labels_type);
    mkdir(example_ann_dir);
end

for a = 1:num_videos
    [~, vidname, ~] = fileparts(annotations(a).name);
    ann_file = fullfile(ann_dir, annotations(a).name);
    load(ann_file);
    
    if(strcmp(labels_type, 'labeled_only'))
        annolist = annolist(find([annolist.is_labeled]));
    end
    
    savepath = [];
    if(save_examples)
        savepath = fullfile(example_ann_dir,  vidname);
        mkdir(savepath);
    end
    
    visualize_annolist(annolist, ROOT_DIR, savepath);
end
    
end