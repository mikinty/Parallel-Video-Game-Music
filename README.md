# Parallel Video Game Music

This repository contains the project files for our Fall 2018 implementation of __Parallel Video Game Music__ for the 15-418 Final Project.

Our project is an implementation of a Markov Model based music generation 
algorithm that utilizes parallelism techniques, including GPU processing and multi-processing, in many steps in order to improve performance. Our final product is able to generate music in live-time, based on custom user settings.

## Directory Structure

- `client`: contains the frontend code for the client that makes requests to the server for music, and plays back generated music to the user.
- `docs`: contains the various writeups involved with this project, including the proposal, checkpoint, and final writeup. These files are also served on the project website.
- `plots`: contains graphing code used to produce visuals for the Results section in the final paper/poster
- `poster`: contains the Powerpoint and PDF versions of the poster
- `src`: contains all the backend code used to implement this project, including preprocessing, training, and music generation.

For more information about each folder, please see their respective READMEs.
