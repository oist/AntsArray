function slp2csv(filename) 
% Step 1: Import SLP file
dset = importSLP(filename);

% Step 2: Flatten the Data
flattenedData = flattenData(dset);

% Step 3: Structure the Data (if this step is necessary for your workflow)
% dset = structureData(dset);

% Step 4: Save the Data for DataFrame Conversion
% csv
csvFileName = fullfile(dset.dir, dset.name+".csv");
writetable(flattenedData, csvFileName);
fprintf(' saved: %s\n', csvFileName);

% panda data frame
% pandaFileName = fullfile(dset.dir, dset.name+".parquet");
% saveDataForPandas(flattenedData, pandaFileName);
end

%%
% SLP Import Function
function dset = importSLP(slpFilePath)
if exist(slpFilePath, 'file') ~= 2
    error("input is not a file. Please select a SLP file.")
else
    fileIdx = 1;
    [dset(fileIdx).dir, dset(fileIdx).name, dset(fileIdx).ext]= fileparts(slpFilePath);
    D = h5info(slpFilePath);
    DatasetNames = {D.Datasets.Name};
    isJson = contains(DatasetNames,'_json');
    dset(fileIdx).tracks = [];
    dset(fileIdx).occupancy_matrix = [];
    dset(fileIdx).tracks_matrix = [];

    % load attribution data
    dset(fileIdx).Attr = jsondecode(D.Groups.Attributes(...
        strcmp({D.Groups.Attributes.Name},'json')).Value);
    if ~isempty(dset(fileIdx).Attr.skeletons)
        [dset(fileIdx).Attr.nodes.('id')] = dset(fileIdx).Attr.skeletons.nodes.id;
    end

    % load each dataset
    for n = 1:numel(DatasetNames)
        if isJson(n)
            dset(fileIdx).tracks.(DatasetNames{n}) = parseJson(...
                h5read(slpFilePath,['/' DatasetNames{n}]), DatasetNames{n});
        else
            dset(fileIdx).tracks.(DatasetNames{n}) = h5read(...
                slpFilePath,['/' DatasetNames{n}]);
        end
    end
    clear D DatasetNames isJson

    % instance data
    dset(fileIdx).nFrame = numel(dset(fileIdx).tracks.frames.frame_idx);
    dset(fileIdx).nAnimals = numel(dset(fileIdx).tracks.instances.instance_id); % N of predicted instances
    dset(fileIdx).nNodes = numel(dset(fileIdx).Attr.nodes);

    fprintf(' imported h5 file: %s\n # of instances: %d (%d frames)\n', ...
        strcat(dset(fileIdx).name, dset(fileIdx).ext), dset(fileIdx).nAnimals, dset(fileIdx).nFrame);
end

    function encoded = parseJson(json,jsontype)
        % encode json text into a matlab handy format
        if ~isempty(json)
            switch jsontype
                case 'tracks_json'
                    % convert tracks_json to num array
                    % exp = '(\d*)';
                    exp = '([)|(])|(")';
                    encoded = split(deblank(regexprep(json, exp, '')), ',');
                    tmp = str2double(encoded); % gives NaN to non-numeric cells
                    encoded(~isnan(tmp)) = num2cell(uint32(tmp(~isnan(tmp))));
                    headers = cell(1,size(encoded,2));
                    headers(1:2) = {'frame_start','track_id'};
                    encoded = array2table(encoded, 'VariableNames', headers);

                    %%% possibe alternative functions for this parts
                    %%% use isstrprop(encoded,'digit'), then apply uint32 to the cols with all([ans{:,1}])

                case {'suggestions_json','videos_json'}
                    encoded = cellfun(@jsondecode, deblank(json));
                otherwise
                    encoded = json;
            end
        else
            encoded = json;
        end
    end
end

function flattenedData = flattenData(dset)
fileIdx = 1;
% flatten frame_instance_xy data to 2D
nInstances = length(dset(fileIdx).tracks.instances.frame_id);
frames_with_instances=dset(fileIdx).tracks.frames.frame_idx + 1;
frameIds = frames_with_instances(dset(fileIdx).tracks.instances.frame_id+1);
pointIdStarts = dset(fileIdx).tracks.instances.point_id_start + 1; % Adjust for 1-based indexing
bodyPointIndices = repmat((1:dset(fileIdx).nNodes)', nInstances, 1);
numRows = length(dset(fileIdx).tracks.pred_points.x);

% Preallocate arrays
instanceIndices = zeros(numRows, 1);
frameIndices = zeros(numRows, 1);

% Populate the arrays
for i = 1:nInstances
    if i < nInstances
        endIndex = pointIdStarts(i+1) - 1;
    else
        endIndex = numRows; % Last instance goes until the end
    end
    instanceIndices(pointIdStarts(i):endIndex) = i;
    frameIndices(pointIdStarts(i):endIndex) = frameIds(i);
end

flattenedData = table( ...
    frameIndices, ...
    instanceIndices, ...
    bodyPointIndices, ...
    dset(fileIdx).tracks.pred_points.x, ...
    dset(fileIdx).tracks.pred_points.y, ...
    dset(fileIdx).tracks.pred_points.score , ...
    'VariableNames', {'Frame', 'Instance', 'Bodypoint', 'X', 'Y', 'Score_node'});
end

function dset = structureData(dset)
fileIdx = 1;

% Preallocate cell arrays for frames and instances
numFrames = dset(fileIdx).nFrame;
numInstances = dset(fileIdx).nAnimals;

% Initialize the cell array for frames
dset(fileIdx).frame_instance_xy = cell(numFrames, 1);

% Adjusting loop to account for MATLAB's 1-based indexing
for frameIdx = 1:numFrames
    % Adjust frameNumber to be 1-based for MATLAB indexing
    idxInstances = dset(fileIdx).tracks.instances.frame_id==(frameIdx-1);
    numInstances = sum(idxInstances);

    % Initialize the cell array for instances within this frame
    instancesCell = cell(sum(idxInstances), 1);
    % instance id for frameIdx
    instance_id = dset(fileIdx).tracks.instances.instance_id(idxInstances);

    for instanceIdx = 1:numInstances
        % Extract x, y index for all nodes of this instance in this frame
        startIndex = dset(fileIdx).tracks.instances.point_id_start(instance_id(instanceIdx) + 1) + 1;
        endIndex = dset(fileIdx).tracks.instances.point_id_end(instance_id(instanceIdx) + 1); % somehow point_id_end is 1-based indexing

        % Extract x and y coordinates for all nodes of this instance in this frame
        xCoords = dset(fileIdx).tracks.pred_points.x(startIndex:endIndex);
        yCoords = dset(fileIdx).tracks.pred_points.y(startIndex:endIndex);
        scoreNode = dset(fileIdx).tracks.pred_points.score(startIndex:endIndex);

        % Store the coordinates in a 2xN matrix (or Nx2 if you prefer) for this instance
        instancesCell{instanceIdx} = [xCoords, yCoords, scoreNode];
    end

    % Assign the instances cell array to the corresponding frame in the main structure
    dset(fileIdx).frame_instance_xy{frameIdx} = instancesCell;
end
end

function saveDataForPandas(flattenedData, parquetFileName)
% For saving to Parquet, ensure the Python environment is set up correctly
% check with 'pyenv' command like:
% pyenv(Version="C:\Users\machi\AppData\Local\Programs\Python\Python39\python.exe")
%
% Initialize an empty Python dictionary
pyDataDict = py.dict();
columnNames = flattenedData.Properties.VariableNames;

% Manually add each column to the Python dictionary
for i = 1:numel(columnNames)
    colData = flattenedData.(columnNames{i});
    % For numeric and logical columns, convert to Python list
    if isnumeric(colData) || islogical(colData)
        pyDataDict{columnNames{i}} = py.list(colData);
    else % For cell arrays of strings, directly convert to Python list
        pyDataDict{columnNames{i}} = py.list(string(colData));
    end
end

% Create the pandas DataFrame
pandasDataFrame = py.pandas.DataFrame(pyDataDict);
pandasDataFrame.to_parquet(parquetFileName);
fprintf(' saved: %s\n', parquetFileName);
end
