% EXAMPLE MATLAB CODE

% -------------------------------------------------------
% 1) Put all the filenames into a cell array
% -------------------------------------------------------
filenames = {
    "X:\ReiterU\Ants\basler\20241109_1\data\cam0_2024-11-09-00-01-27_cam01_000.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam0_2024-11-09-00-01-27_cam01_001.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam0_2024-11-09-00-01-27_cam01_002.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam0_2024-11-09-00-01-27_cam01_003.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam0_2024-11-09-00-01-27_cam01_004.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam0_2024-11-09-00-01-27_cam01_frame_counts.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam0_2024-11-09-00-01-50_cam18_000.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam0_2024-11-09-00-01-50_cam18_001.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam0_2024-11-09-00-01-50_cam18_002.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam0_2024-11-09-00-01-50_cam18_003.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam0_2024-11-09-00-01-50_cam18_004.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam0_2024-11-09-00-01-50_cam18_frame_counts.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam0_2024-11-09-00-02-12_cam09_000.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam0_2024-11-09-00-02-12_cam09_001.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam0_2024-11-09-00-02-12_cam09_002.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam0_2024-11-09-00-02-12_cam09_003.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam0_2024-11-09-00-02-12_cam09_004.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam0_2024-11-09-00-02-12_cam09_frame_counts.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam1_2024-11-09-00-01-26_cam02_000.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam1_2024-11-09-00-01-26_cam02_001.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam1_2024-11-09-00-01-26_cam02_002.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam1_2024-11-09-00-01-26_cam02_004.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam1_2024-11-09-00-01-26_cam02_frame_counts.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam1_2024-11-09-00-01-49_cam19_000.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam1_2024-11-09-00-01-49_cam19_001.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam1_2024-11-09-00-01-49_cam19_002.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam1_2024-11-09-00-01-49_cam19_003.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam1_2024-11-09-00-01-49_cam19_004.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam1_2024-11-09-00-01-49_cam19_frame_counts.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam1_2024-11-09-00-02-11_cam10_000.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam1_2024-11-09-00-02-11_cam10_001.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam1_2024-11-09-00-02-11_cam10_002.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam1_2024-11-09-00-02-11_cam10_003.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam1_2024-11-09-00-02-11_cam10_004.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam1_2024-11-09-00-02-11_cam10_frame_counts.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam2_2024-11-09-00-01-27_cam03_000.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam2_2024-11-09-00-01-27_cam03_001.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam2_2024-11-09-00-01-27_cam03_002.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam2_2024-11-09-00-01-27_cam03_003.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam2_2024-11-09-00-01-27_cam03_004.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam2_2024-11-09-00-01-27_cam03_frame_counts.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam2_2024-11-09-00-01-51_cam20_000.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam2_2024-11-09-00-01-51_cam20_001.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam2_2024-11-09-00-01-51_cam20_002.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam2_2024-11-09-00-01-51_cam20_003.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam2_2024-11-09-00-01-51_cam20_004.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam2_2024-11-09-00-01-51_cam20_frame_counts.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam2_2024-11-09-00-02-12_cam11_000.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam2_2024-11-09-00-02-12_cam11_001.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam2_2024-11-09-00-02-12_cam11_002.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam2_2024-11-09-00-02-12_cam11_003.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam2_2024-11-09-00-02-12_cam11_004.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam2_2024-11-09-00-02-12_cam11_frame_counts.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam3_2024-11-09-00-01-27_cam04_000.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam3_2024-11-09-00-01-27_cam04_001.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam3_2024-11-09-00-01-27_cam04_002.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam3_2024-11-09-00-01-27_cam04_003.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam3_2024-11-09-00-01-27_cam04_004.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam3_2024-11-09-00-01-27_cam04_frame_counts.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam3_2024-11-09-00-01-50_cam21_000.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam3_2024-11-09-00-01-50_cam21_001.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam3_2024-11-09-00-01-50_cam21_002.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam3_2024-11-09-00-01-50_cam21_003.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam3_2024-11-09-00-01-50_cam21_004.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam3_2024-11-09-00-01-50_cam21_frame_counts.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam3_2024-11-09-00-02-13_cam12_000.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam3_2024-11-09-00-02-13_cam12_001.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam3_2024-11-09-00-02-13_cam12_002.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam3_2024-11-09-00-02-13_cam12_003.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam3_2024-11-09-00-02-13_cam12_004.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam3_2024-11-09-00-02-13_cam12_frame_counts.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam4_2024-11-09-00-01-27_cam05_000.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam4_2024-11-09-00-01-27_cam05_001.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam4_2024-11-09-00-01-27_cam05_002.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam4_2024-11-09-00-01-27_cam05_003.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam4_2024-11-09-00-01-27_cam05_004.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam4_2024-11-09-00-01-27_cam05_frame_counts.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam4_2024-11-09-00-01-51_cam22_000.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam4_2024-11-09-00-01-51_cam22_001.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam4_2024-11-09-00-01-51_cam22_002.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam4_2024-11-09-00-01-51_cam22_003.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam4_2024-11-09-00-01-51_cam22_004.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam4_2024-11-09-00-01-51_cam22_frame_counts.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam4_2024-11-09-00-02-12_cam13_000.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam4_2024-11-09-00-02-12_cam13_001.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam4_2024-11-09-00-02-12_cam13_002.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam4_2024-11-09-00-02-12_cam13_003.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam4_2024-11-09-00-02-12_cam13_004.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam4_2024-11-09-00-02-12_cam13_frame_counts.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam5_2024-11-09-00-01-26_cam06_000.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam5_2024-11-09-00-01-26_cam06_001.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam5_2024-11-09-00-01-26_cam06_002.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam5_2024-11-09-00-01-26_cam06_003.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam5_2024-11-09-00-01-26_cam06_004.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam5_2024-11-09-00-01-26_cam06_frame_counts.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam5_2024-11-09-00-01-50_cam23_000.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam5_2024-11-09-00-01-50_cam23_001.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam5_2024-11-09-00-01-50_cam23_002.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam5_2024-11-09-00-01-50_cam23_003.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam5_2024-11-09-00-01-50_cam23_004.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam5_2024-11-09-00-01-50_cam23_frame_counts.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam5_2024-11-09-00-02-12_cam14_000.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam5_2024-11-09-00-02-12_cam14_001.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam5_2024-11-09-00-02-12_cam14_002.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam5_2024-11-09-00-02-12_cam14_003.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam5_2024-11-09-00-02-12_cam14_004.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam5_2024-11-09-00-02-12_cam14_frame_counts.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam6_2024-11-09-00-01-26_cam07_000.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam6_2024-11-09-00-01-26_cam07_001.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam6_2024-11-09-00-01-26_cam07_002.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam6_2024-11-09-00-01-26_cam07_003.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam6_2024-11-09-00-01-26_cam07_004.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam6_2024-11-09-00-01-26_cam07_frame_counts.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam6_2024-11-09-00-01-51_cam24_000.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam6_2024-11-09-00-01-51_cam24_001.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam6_2024-11-09-00-01-51_cam24_002.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam6_2024-11-09-00-01-51_cam24_003.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam6_2024-11-09-00-01-51_cam24_004.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam6_2024-11-09-00-01-51_cam24_frame_counts.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam6_2024-11-09-00-02-11_cam15_000.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam6_2024-11-09-00-02-11_cam15_001.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam6_2024-11-09-00-02-11_cam15_002.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam6_2024-11-09-00-02-11_cam15_003.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam6_2024-11-09-00-02-11_cam15_004.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam6_2024-11-09-00-02-11_cam15_frame_counts.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam7_2024-11-09-00-01-26_cam08_000.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam7_2024-11-09-00-01-26_cam08_001.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam7_2024-11-09-00-01-26_cam08_002.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam7_2024-11-09-00-01-26_cam08_003.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam7_2024-11-09-00-01-26_cam08_004.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam7_2024-11-09-00-01-26_cam08_frame_counts.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam7_2024-11-09-00-01-50_cam25_000.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam7_2024-11-09-00-01-50_cam25_001.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam7_2024-11-09-00-01-50_cam25_002.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam7_2024-11-09-00-01-50_cam25_003.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam7_2024-11-09-00-01-50_cam25_004.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam7_2024-11-09-00-01-50_cam25_frame_counts.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam7_2024-11-09-00-02-12_cam16_000.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam7_2024-11-09-00-02-12_cam16_001.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam7_2024-11-09-00-02-12_cam16_002.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam7_2024-11-09-00-02-12_cam16_003.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam7_2024-11-09-00-02-12_cam16_004.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam7_2024-11-09-00-02-12_cam16_frame_counts.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam8_2024-11-09-00-02-12_cam17_000.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam8_2024-11-09-00-02-12_cam17_001.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam8_2024-11-09-00-02-12_cam17_002.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam8_2024-11-09-00-02-12_cam17_003.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam8_2024-11-09-00-02-12_cam17_004.csv"
"X:\ReiterU\Ants\basler\20241109_1\data\cam8_2024-11-09-00-02-12_cam17_frame_counts.csv"
};

% -------------------------------------------------------
% 2) Extract the "secondary" camXX or "global" from each filename
% -------------------------------------------------------
nFiles      = numel(filenames);
camIDCell   = cell(nFiles,1);    % cell to store the extracted ID string
camIDNumeric= nan(nFiles,1);     % numeric ID if parseable (e.g., cam09 -> 9)

for k = 1:nFiles
    thisFile = filenames{k};
    
    % Use a regular expression to grab what's after "_cam" and before the next underscore
    % Example match: "_cam18_"
    tokens = regexp(thisFile, '_cam(\d{2})_', 'tokens', 'once');
    
    % If that doesn't match, maybe we check for "_global_"
    if ~isempty(tokens)
        % Convert '18' -> 18, etc.
        camIDNumeric(k) = str2double(tokens{1});
        camIDCell{k}    = ['cam' tokens{1}];
    else
        % Check if file contains '_global_' 
        tokensGlobal = regexp(thisFile, '_global_', 'match', 'once');
        if ~isempty(tokensGlobal)
            camIDCell{k}    = 'global';
            camIDNumeric(k) = -1;  % or any sentinel value
        else
            camIDCell{k}    = 'UNKNOWN'; 
            camIDNumeric(k) = NaN; 
        end
    end
end

% -------------------------------------------------------
% 3) Sort by the numeric camera ID (ascending)
%    (Here, global = -1 will come before 0 or any real cams.)
% -------------------------------------------------------
[~, sortIdx] = sort(camIDNumeric);

sortedFilenames = filenames(sortIdx);
sortedCamIDCell = camIDCell(sortIdx);

% -------------------------------------------------------
% 4) Display or group the results
% -------------------------------------------------------
disp('*** Sorted filenames by secondary camID ***');
for i = 1:nFiles
    fprintf('%02d) %s  [secondary=%s]\n', i, sortedFilenames{i}, sortedCamIDCell{i});
end

% ---------------
% OPTIONAL: Grouping
% ---------------
uniqueIDs = unique(sortedCamIDCell, 'stable');
for ui = 1:numel(uniqueIDs)
    theID = uniqueIDs{ui};
    disp(['------ ' theID ' ------']);
    idxGroup = strcmp(sortedCamIDCell, theID);
    disp(sortedFilenames(idxGroup));
end
