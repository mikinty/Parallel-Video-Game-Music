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

export const CHORD_START = 2;

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
export const NOTE_DURATIONS = [
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
  voices: [0, 1, 2, 0, 0, 0, 0, 0, 0, 0],
  tempo: 200
};

export const SETTINGS_JOURNEY = {
  name: 'journey',
  mood: 0,
  voices: [0, 1, 2, 2, 0, 0, 0, 0, 0, 0],
  tempo: 220
};

export const SETTINGS_SORROW = {
  name: 'sorrow',
  mood: 1,
  voices: [0, 1, 1, 2, 0, 0, 0, 0, 0, 0],
  tempo: 150
};

export const SETTINGS_BATTLE= {
  name: 'battle',
  mood: 0,
  voices: [0, 0, 1, 1, 2, 2, 0, 0, 0, 0],
  tempo: 300
};