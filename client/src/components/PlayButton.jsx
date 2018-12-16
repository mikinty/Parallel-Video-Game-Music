import React from 'react';
import Tone from 'tone';
import * as SECRET from '../KEY';
import * as CONST from '../CONST';
import * as DATA from '../DATA';

// Do Tone js setup here

// 10 voice synth
const SYNTHS = [
  [
    new Tone.Synth(),
    new Tone.Synth(),
    new Tone.Synth() 
  ],
  [
    new Tone.Synth(),
    new Tone.Synth(),
    new Tone.Synth()
  ],
  [
    new Tone.Synth(),
    new Tone.Synth(),
    new Tone.Synth()
  ],
  [
    new Tone.Synth(),
    new Tone.Synth(),
    new Tone.Synth()
  ],
  [
    new Tone.Sampler({
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
    }, 
    {
			'release' : 1,
			'baseUrl' : './samples/piano/'
		}),
    new Tone.Synth(),
    new Tone.Synth()
  ],
  [
    new Tone.Synth(),
    new Tone.Synth(),
    new Tone.Synth()
  ],
  [
    new Tone.Synth(),
    new Tone.Synth(),
    new Tone.Synth()
  ],
  [
    new Tone.Synth(),
    new Tone.Synth(),
    new Tone.Synth()
  ],
  [
    new Tone.Synth(),
    new Tone.Synth(),
    new Tone.Synth()
  ],
  [
    new Tone.Synth(),
    new Tone.Synth(),
    new Tone.Synth()
  ]
];

// connect to speakers
SYNTHS.forEach(s => {
  s[0].toMaster();
  s[1].toMaster();
  s[2].toMaster(); 
});

Tone.Transport.bpm.value = 160;

// const transport = new Tone.Transport();
const ws = new WebSocket('ws://' + SECRET.SECRET_ADDRESS);

/**
 * Parses an integer encoding a chord.
 * There is a CHORD_OFFSET that needs to be subtracted.
 * The highest pitched note is the most significant. 
 * 
 * @param {int} chord 
 * 
 * @returns an array of describing the notes the chord maps to
 */
function parseChord (chord) {
  // Convert to base 12
  let temp = chord - CONST.CHORD_OFFSET;
  temp = temp.toString(CONST.DISTINCT_NOTES);

  let notes = [];

  // first iteration
  let prevNote = CONST.NOTE_TYPES[parseInt(temp[temp.length - 1], CONST.DISTINCT_NOTES)];
  let octave = CONST.CHORD_START;

  notes.push(prevNote + octave);

  for (let i = temp.length - 2; i >= 0; i--) {
    if (parseInt(temp[i]) < prevNote) {
      octave++;  
    }

    prevNote = CONST.NOTE_TYPES[parseInt(temp[i], CONST.DISTINCT_NOTES)];

    notes.push(prevNote + octave);
  }

  return notes;
}

/**
 * Plays the notes specified by notes
 * 
 * @param {array array (note * duration)} notes 
 */
function playNotes (notes) {
  // reset transport
  Tone.Transport.stop();
  Tone.Transport.clear(0);
  Tone.Transport.seconds = 0;

  // parse through the notes we are getting  
  for (let j = 4; j < 5; j++) {
    var currTime = 0;

    for (let i = 0; i < notes[j].length; i++) { 
      console.log(i, notes[j][i]);

      let note = notes[j][i][0];
      let duration = notes[j][i][1];

      // Schedule the music we just received
      if (note == CONST.REST_NOTE) {
        // rest
        Tone.Transport.schedule((time) => {
          console.log('rest');
          SYNTHS[j][0].triggerRelease(time);
        }, currTime);
      } else if (note < CONST.REST_NOTE) {
        // note
        Tone.Transport.schedule((time) => {
          /*
          console.log(
            'playing',
            CONST.NOTE_MAPPINGS[note],
            CONST.NOTE_DURATIONS[duration]
          );
          */

          SYNTHS[j][0].triggerAttackRelease (
            CONST.NOTE_MAPPINGS[note],
            CONST.NOTE_DURATIONS[duration],
            time
          );
        }, currTime);
      } else {
        // chord
        let tempNotes = parseChord(note);

        for (let i = 0; i < tempNotes.length; i++) {
          Tone.Transport.schedule((time) => {
            console.log('chords playing', tempNotes[i], CONST.NOTE_DURATIONS[duration]);  

            SYNTHS[j][i].triggerAttackRelease (
              tempNotes[i],
              CONST.NOTE_DURATIONS[duration],
              time
            );
          }, currTime);
        }
      }

      currTime = currTime + CONST.NOTE_DURATIONS[notes[j][i][1]];
    }
  }

  console.log('starting', Tone.Transport);
  Tone.Transport.start();
}

// listen to websocket events
ws.onmessage = (event) => {
  console.log(event);

  // [[[a, b]], [[c, d]], [] ]
  var notes = JSON.parse(event.data).notes;

  playNotes(notes);

}

export default class PlayButton extends React.Component {
  constructor () {
    super();
  }
  

  // Called when we click play
  handleClick () {
    /*
    ws.send(JSON.stringify({
      request: CONST.START_MUSIC_REQ,
      data: 'TODD MOWRY IS THE BEST'
    }));
    */

    playNotes(DATA.testNotes);
  }

  render() {
    return (
      <div className='playBut' onClick={this.handleClick}>
        <svg version="1.1"
          xmlns="http://www.w3.org/2000/svg"
          x="0px" y="0px" width="120px" height="120px" viewBox="0 0 213.7 213.7" enableBackground="new 0 0 213.7 213.7"
          space="preserve">
        
          <polygon className='triangle' id="XMLID_18_" fill="none" strokeWidth="7" strokeLinecap="round" strokeLinejoin="round" strokeMiterlimit="10" points="73.5,62.5 148.5,105.8 73.5,149.1 "/>
            
          <circle className='circle' id="XMLID_17_" fill="none"  strokeWidth="7" strokeLinecap="round" strokeLinejoin="round" strokeMiterlimit="10" cx="106.8" cy="106.8" r="103.3"/>
        </svg>
      </div>
    );
  }
}