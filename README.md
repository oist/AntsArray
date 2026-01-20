# AntsArray

> This project can optionally use **uv** for dependency management: `pyproject.toml` declares dependencies, and `uv.lock` enables reproducible installs across Linux and Windows (`uv sync`).


For colony data, after inference, map_combine combines multicam data into one file/per chunk for aruco and sleap. Uses the aruco h5 to encode number of frames. Write into a single folder for all big videos that should be processed together. aruco_track (need to update), or the slurm version submit_chunk.sh runs tracking per chunk, saves chunk tracking data to parquet files. 

Downstream analysis so far first generates single data over chunks with single_ant_over_chunks.py, and analyses them. Now that multiple video files all live in a single directory, I can move across them for single ants to get one ant over long times. 

for single ant data: single_ant.py generates one file per arena in a dataset, with arenas given by a segmentation image. By saving all datasets into a common folder, I can unify downstream analysis with the colony data. single_ant_over_chunks.py can be run to generate a per_track file for each ant over datasets.


I don't think single_ant_over_chunks.py utilizes the num_frames field saved from the aruco.h5 yet. Need to make sure that is propagated to tracking parquet file, and to add it to the pipeline for processing single ants.

check in on the issue of variable chunk lengths and syncing.

I dont think the angle measure for sleep determination is working so well. 20251218 cam 12 has a cocoon and it looks like the ant is still and sleeping a lot more than the angle measure pulls out. Need to refine or change, for now just use speed.

need to deal with gap in between recordings. When generating single tracks over datasets need to use filename timestamp and fps to pad the frame locations like it was a continuous recording with some missing data. Make a single file vector with data present/not boolian. 


for actual biology, should look at changing sleep definition to a time when absolutely still. Then look at how sleeping ants react to taps of different strength compared to waking ants.
