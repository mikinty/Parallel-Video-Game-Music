export const TITLE = 'Parallel Video Game Music';

export const NOTE_TYPES = [
  'C',
  'C#',
  'D',
  'D#',
  'E',
  'F',
  'F#',
  'G',
  'G#',
  'A',
  'A#',
  'B'
];

export const CHORD_START = 3;

// Music constants mapped to 0 - 71
export const NOTE_MAPPINGS = [
  // Octave 1
  'C1',
  'C#1',
  'D1',
  'D#1',
  'E1',
  'F1',
  'F#1',
  'G1',
  'G#1',
  'A1',
  'A#1',
  'B1',

  // Octave 2
  'C2',
  'C#2',
  'D2',
  'D#2',
  'E2',
  'F2',
  'F#2',
  'G2',
  'G#2',
  'A2',
  'A#2',
  'B2',

  // Octave 3
  'C3',
  'C#3',
  'D3',
  'D#3',
  'E3',
  'F3',
  'F#3',
  'G3',
  'G#3',
  'A3',
  'A#3',
  'B3',

  // Octave 4
  'C4',
  'C#4',
  'D4',
  'D#4',
  'E4',
  'F4',
  'F#4',
  'G4',
  'G#4',
  'A4',
  'A#4',
  'B4',

  // Octave 5
  'C5',
  'C#5',
  'D5',
  'D#5',
  'E5',
  'F5',
  'F#5',
  'G5',
  'G#5',
  'A5',
  'A#5',
  'B5',

  // Octave 6
  'C6',
  'C#6',
  'D6',
  'D#6',
  'E6',
  'F6',
  'F#6',
  'G6',
  'G#6',
  'A6',
  'A#6',
  'B6'
];

export const DISTINCT_NOTES = 12;
export const REST_NOTE = 72;
export const CHORD_OFFSET = 101;

// Note durations mapped to 0 - 14
export const NOTE_DURATIONS_RAW = [
  0.083, // 1
  0.167, // 2
  0.250, // 3
  0.333, // 4
  0.500, // 6
  0.667, // 8
  0.750, // 9
  1.000, // 12
  1.333, // 16
  1.500, // 18
  1.750, // 21
  2.000, // 24
  3.000, // 36
  4.000, // 48
  8.000  // 96
];

export const NOTE_DURATIONS = [
  '1i', // 1
  '2i', // 2
  '3i', // 3
  '4i', // 4
  '6i', // 6
  '8i', // 8
  '9i', // 9
  '12i', // 12
  '16i', // 16
  '18i', // 18
  '21i', // 21
  '24i', // 24
  '36i', // 36
  '48i', // 48
  '96i'  // 96
];

// Server commands
export const START_MUSIC_REQ = 'START_MUSIC';

/**
 * Preset settings to give to server. 
 * 
 * For voices, must be length 10, and:
 * 0 = chord, 1 = bass, 2 = soprano, -1 = silent
 * 
 * For mood: 0 major / 1 minor
 */
export const SETTINGS_BASIC = {
  name: 'basic',
  mood: 0,
  voices: [0, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  measures: 5,
  tempo: 11
};

export const SETTINGS_JOURNEY = {
  name: 'journey',
  mood: 0,
  voices: [-1, -1, -1, -1, 2, 2, -1, -1, -1, -1],
  measures: 10,
  tempo: 10
};

export const SETTINGS_SORROW = {
  name: 'sorrow',
  mood: 1,
  voices: [0, 1, -1, 2, -1, -1, -1, -1, -1, -1],
  measures: 8,
  tempo: 4
};

export const SETTINGS_BATTLE = {
  name: 'battle',
  mood: 1,
  voices: [-1, -1, -1, -1, -1, -1, -1, 0, -1, -1],
  measures: 10,
  tempo: 13
};

export const VOLUME = -2;
export const ADSR = {
  A: 0.05,
  D: 0.5,
  S: 0.2,
  R: 1
};

export const VOLUME_CHORD = -1;

export const LOWPASS_FREQ = 3000;

export const HIGHPASS_FREQ = 0;

/** 
 * A bunch of configuration files for Tone.Sampler
 * Requires samples to be placed in the dist/ folder
 */
export const PIANO_SETTINGS_FILES = {
  'A0': 'A0.[mp3|ogg]',
  'A1': 'A1.[mp3|ogg]',
  'A2': 'A2.[mp3|ogg]',
  'A3': 'A3.[mp3|ogg]',
  'A4': 'A4.[mp3|ogg]',
  'A5': 'A5.[mp3|ogg]',
  'A6': 'A6.[mp3|ogg]',
  'A#0': 'As0.[mp3|ogg]',
  'A#1': 'As1.[mp3|ogg]',
  'A#2': 'As2.[mp3|ogg]',
  'A#3': 'As3.[mp3|ogg]',
  'A#4': 'As4.[mp3|ogg]',
  'A#5': 'As5.[mp3|ogg]',
  'A#6': 'As6.[mp3|ogg]',
  'B0': 'B0.[mp3|ogg]',
  'B1': 'B1.[mp3|ogg]',
  'B2': 'B2.[mp3|ogg]',
  'B3': 'B3.[mp3|ogg]',
  'B4': 'B4.[mp3|ogg]',
  'B5': 'B5.[mp3|ogg]',
  'B6': 'B6.[mp3|ogg]',
  'C0': 'C0.[mp3|ogg]',
  'C1': 'C1.[mp3|ogg]',
  'C2': 'C2.[mp3|ogg]',
  'C3': 'C3.[mp3|ogg]',
  'C4': 'C4.[mp3|ogg]',
  'C5': 'C5.[mp3|ogg]',
  'C6': 'C6.[mp3|ogg]',
  'C7': 'C7.[mp3|ogg]',
  'C#0': 'Cs0.[mp3|ogg]',
  'C#1': 'Cs1.[mp3|ogg]',
  'C#2': 'Cs2.[mp3|ogg]',
  'C#3': 'Cs3.[mp3|ogg]',
  'C#4': 'Cs4.[mp3|ogg]',
  'C#5': 'Cs5.[mp3|ogg]',
  'C#6': 'Cs6.[mp3|ogg]',
  'D0': 'D0.[mp3|ogg]',
  'D1': 'D1.[mp3|ogg]',
  'D2': 'D2.[mp3|ogg]',
  'D3': 'D3.[mp3|ogg]',
  'D4': 'D4.[mp3|ogg]',
  'D5': 'D5.[mp3|ogg]',
  'D6': 'D6.[mp3|ogg]',
  'D#0': 'Ds0.[mp3|ogg]',
  'D#1': 'Ds1.[mp3|ogg]',
  'D#2': 'Ds2.[mp3|ogg]',
  'D#3': 'Ds3.[mp3|ogg]',
  'D#4': 'Ds4.[mp3|ogg]',
  'D#5': 'Ds5.[mp3|ogg]',
  'D#6': 'Ds6.[mp3|ogg]',
  'E0': 'E0.[mp3|ogg]',
  'E1': 'E1.[mp3|ogg]',
  'E2': 'E2.[mp3|ogg]',
  'E3': 'E3.[mp3|ogg]',
  'E4': 'E4.[mp3|ogg]',
  'E5': 'E5.[mp3|ogg]',
  'E6': 'E6.[mp3|ogg]',
  'F0': 'F0.[mp3|ogg]',
  'F1': 'F1.[mp3|ogg]',
  'F2': 'F2.[mp3|ogg]',
  'F3': 'F3.[mp3|ogg]',
  'F4': 'F4.[mp3|ogg]',
  'F5': 'F5.[mp3|ogg]',
  'F6': 'F6.[mp3|ogg]',
  'F#0': 'Fs0.[mp3|ogg]',
  'F#1': 'Fs1.[mp3|ogg]',
  'F#2': 'Fs2.[mp3|ogg]',
  'F#3': 'Fs3.[mp3|ogg]',
  'F#4': 'Fs4.[mp3|ogg]',
  'F#5': 'Fs5.[mp3|ogg]',
  'F#6': 'Fs6.[mp3|ogg]',
  'G0': 'G0.[mp3|ogg]',
  'G1': 'G1.[mp3|ogg]',
  'G2': 'G2.[mp3|ogg]',
  'G3': 'G3.[mp3|ogg]',
  'G4': 'G4.[mp3|ogg]',
  'G5': 'G5.[mp3|ogg]',
  'G6': 'G6.[mp3|ogg]',
  'G#0': 'Gs0.[mp3|ogg]',
  'G#1': 'Gs1.[mp3|ogg]',
  'G#2': 'Gs2.[mp3|ogg]',
  'G#3': 'Gs3.[mp3|ogg]',
  'G#4': 'Gs4.[mp3|ogg]',
  'G#5': 'Gs5.[mp3|ogg]',
  'G#6': 'Gs6.[mp3|ogg]'
}

export const PIANO_SETTINGS_URL = {
  'release' : 1,
  'baseUrl' : './samples/piano/'
};

export const VIOLIN_SETTINGS_FILES = {
  'A3': 'A3.[mp3|ogg]',
  'A4': 'A4.[mp3|ogg]',
  'A5': 'A5.[mp3|ogg]',
  'A6': 'A6.[mp3|ogg]',
  'C4': 'C4.[mp3|ogg]',
  'C5': 'C5.[mp3|ogg]',
  'C6': 'C6.[mp3|ogg]',
  'C7': 'C7.[mp3|ogg]',
  'E4': 'E4.[mp3|ogg]',
  'E5': 'E5.[mp3|ogg]',
  'E6': 'E6.[mp3|ogg]',
  'G4': 'G4.[mp3|ogg]',
  'G5': 'G5.[mp3|ogg]',
  'G6': 'G6.[mp3|ogg]'
};

export const VIOLIN_SETTINGS_URL = {
  'release' : 1,
  'baseUrl' : './samples/violin/'
};


export const XYLO_SETTINGS_FILES = {
  'C7': 'C7.[mp3|ogg]',
  'G3': 'G3.[mp3|ogg]',
  'G4': 'G4.[mp3|ogg]',
  'G5': 'G5.[mp3|ogg]',
  'G6': 'G6.[mp3|ogg]',
  'C4': 'C4.[mp3|ogg]',
  'C5': 'C5.[mp3|ogg]',
  'C6': 'C6.[mp3|ogg]'
};

export const XYLO_SETTINGS_URL = {
  'release' : 1,
  'baseUrl' : './samples/xylophone/'
};

export const TRUMPET_SETTINGS_FILES = {
  'C5': 'C5.[mp3|ogg]',
  'D4': 'D4.[mp3|ogg]',
  'D#3': 'Ds3.[mp3|ogg]',
  'F2': 'F2.[mp3|ogg]',
  'F3': 'F3.[mp3|ogg]',
  'F4': 'F4.[mp3|ogg]',
  'G3': 'G3.[mp3|ogg]',
  'A2': 'A2.[mp3|ogg]',
  'A4': 'A4.[mp3|ogg]',
  'A#3': 'As3.[mp3|ogg]',
  'C3': 'C3.[mp3|ogg]'
};

export const TRUMPET_SETTINGS_URL = {
  'release' : 1,
  'baseUrl' : './samples/trumpet/'
};

export const TUBA_SETTINGS_FILES = {
  'A#1': 'As1.[mp3|ogg]',
  'A#2': 'As2.[mp3|ogg]',
  'D2': 'D2.[mp3|ogg]',
  'D3': 'D3.[mp3|ogg]',
  'D#1': 'Ds1.[mp3|ogg]',
  'F0': 'F0.[mp3|ogg]',
  'F1': 'F1.[mp3|ogg]',
  'F2': 'F2.[mp3|ogg]',
  'A#0': 'As0.[mp3|ogg]'
};

export const TUBA_SETTINGS_URL = {
  'release' : 1,
  'baseUrl' : './samples/tuba/'
};