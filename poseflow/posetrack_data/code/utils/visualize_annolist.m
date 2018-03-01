function visualize_annolist(annolist, path_to_images, savepath)

    if(nargin < 3)
        savepath = [];
    end

    total_persons = get_persons_count(annolist);
    num_frames  = length(annolist);

    cmap = hsv(total_persons);  
    edges = [1 2; 2 3; 3 9; 4 10; 4 5; 5 6; 7 8; 8 9; 10 11; 11 12; 9 13; 10 13; 13 14; 14 15 ];
    num_joints  = 15;
    marker_size = 6;
    line_width  = 3;

    figure(101); clf; axis equal;
    for f = 1:num_frames
        
        image_fn = annolist(f).image.name;
        image = imread(fullfile(path_to_images, image_fn));
        imshow(image); hold on;
        
        % draw ignore regions
        if(isfield(annolist(f), 'ignore_regions'))
            ignore_regions = annolist(f).ignore_regions;
            num_ignore = length(ignore_regions);
            for p = 1:num_ignore
                if(isfield(ignore_regions(p), 'point'))
                    num_points = length(ignore_regions(p).point);
                    ir_points  = get_points(ignore_regions(p), num_points);
                    idx  = ~isnan(ir_points(:,1));
                    ir_points = ir_points(idx,:);
                    h = patch(ir_points(:,1), ir_points(:,2), [1,0,0]);
                    set(h,'FaceAlpha',0.5);
                    set(h,'EdgeAlpha',0);
                end
            end
        end
        
        % draw poses
        annorect    = annolist(f).annorect;
        num_persons = length(annorect);
        for p = 1:num_persons
            person_color = cmap(mod(p-1,size(cmap,1))+1,:);
            if(isfield(annorect(p), 'annopoints'))
                annopoints = annorect(p).annopoints;
                joints = get_points(annopoints, num_joints);
                draw_sticks(joints, edges, person_color, line_width);
                draw_joints(joints, marker_size, person_color);
            
                head_rect  = [annorect(p).x1,  annorect(p).y1, ... 
                          annorect(p).x2,  annorect(p).y2];
                draw_head_rect(head_rect, person_color, line_width);
            end
        end
        
        if(~isempty(savepath))
            [~,name,~] = fileparts(image_fn);
            save_image_path = fullfile(savepath, [name, '.jpg']);
            F = getframe;
            F = imresize(F.cdata, [size(image,1), size(image,2)]);
            imwrite(F, save_image_path, 'Quality', 90);
        end
        hold off;
        pause(0.001);
    end

end

function draw_head_rect(rect, person_color, line_width)
    x = rect(1);
    y = rect(2);
    width = rect(3)-rect(1);
    height = rect(4)-rect(2);
    rectangle('Position', [x,y,width,height], ...
        'linewidth',line_width, 'EdgeColor', person_color);
end

function draw_joints(joints, marker_size, person_color)
colors = {'b','y','r','g','m','c'};
for j=1:size(joints,1)
    pos = joints(j,:);
    cp = colors{mod(j-1,length(colors))+1};
    plot(pos(:,1),pos(:,2),[cp 'o'], 'MarkerSize',marker_size, ...
         'MarkerEdgeColor', person_color, ...
         'MarkerFaceColor', cp,  ...
         'LineWidth', 1.5);
end
end


function draw_sticks(joints, edges, person_color, line_width)
%draw sticks
set(gcf,'GraphicsSmoothing','on')
for i = 1:size(edges,1)
    pos1 = joints(edges(i,1), :);
    pos2 = joints(edges(i,2), :);
    if (~isnan(pos1(1)) && ~isnan(pos2(1)))
        plot([pos1(1);pos2(1)],[pos1(2);pos2(2)],'-', 'Color', ...
                    person_color ,'linewidth',line_width, ... 
                    'LineSmoothing','on');
    end
end
end


function points = get_points(annopoints, num_points)
    points = NaN(num_points, 2);
    
    if(isfield(annopoints, 'point'))
        ids  = [annopoints.point.id]+1;
        x = [annopoints.point.x]';
        y = [annopoints.point.y]';
        points(ids,:) = [x, y];
    end
end


function person_count = get_persons_count(annolist)
    person_count = -1;
    for f = 1:length(annolist)
        annorect = annolist(f).annorect;
        if(isfield(annorect, 'track_id'))
            person_count = max(person_count, max([annorect.track_id]));
        end
    end
    person_count = person_count + 1;
end

