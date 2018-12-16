# Parallel Video Game Music

This folder contains the client implementation of the project. We use `yarn` for package management, `webpack` for building, `React` as the framework, and `Tone.js` for generating sounds.

## Install
```
yarn install
```

Since this will install around 100 packages, this may take several minutes.

## Run
```
yarn start
```

Webpack will let you know where the project is served. By default, this should be `localhost:8080`

## Instrumentation

In order to have real instruments instead of synthesizers provided `Tone.js`, you have to install sample audio files yourself.
They are omitted from this repo for space considerations, as they easily exceed 2 GB in all. To install instruments, add 
```
sample/instrument/*.mp3
```
under `dist/`