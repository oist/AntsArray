function batch_slp2csv(slp_dir_path)
if exist(slp_dir_path, 'dir') ~= 7
    error("input is not a directory. Please select a directory containing SLP files.")
else
    % Main
    slpFiles = natsortfiles(dir(slp_dir_path));
    slpFiles = slpFiles(endsWith({slpFiles.name},{'.slp'}));
    slpFiles = fullfile({slpFiles.folder}', {slpFiles.name}');
    
    fprintf("Processing %d slp files in %s\n\n",numel(slpFiles), slp_dir_path)

    for m = 1:numel(slpFiles)
        slp2csv(slpFiles{m})
    end

    fprintf("\n%d slp files are converted.\n", numel(slpFiles)) 
end
end
