# Music Generation on the Server Side

This directory contains the files for the server in the music
generation step.

Files in this directory include:

- `convert.py`: converts the text matrices produced by training into `numpy` arrays
- `major/minor*.pkl`: major and minor transition matrices used as the Markov Model for generating notes
- `main(Seq).py`: the main script for starting the server. Uses websockets to communicate with the client.
- `musicGen(Parallel/Seq).py`: the parallel/sequential implementations of music generation
- `CONST_SERVER.py`: contains the constant definitions for the server code

## Usage

In order to run the server, you need `sudo` access. You don't want to do 
this directly through `sudo python` because for some reason that messes
up with the `anaconda` Python pathing. So you instead want to do:

```
> sudo -s # switch into sudo user mode

> python main(Seq).py
```

The reason for sudo access is because we are serving a program on the 
`0.0.0.0` port, which goes to public access.
