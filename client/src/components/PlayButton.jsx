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
    new Tone.Synth({
      oscillator: {
        type : 'sine',
        volume: -6
      },
      envelope : {
        attack : 0.05,
        decay : 0.25,
        sustain : 0.5,
        release : 1,
      }
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

Tone.Transport.bpm.value = 220;

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
  for (let j = 3; j < 5; j++) {
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
  console.log('Received', event);

  var notes = JSON.parse(event.data).notes;

  playNotes(notes);
}

export default class PlayButton extends React.Component {
  constructor (props) {
    super(props);

    // ES6 requires binding
    this.handleClick = this.handleClick.bind(this);
  }

  // Called when we click play
  // Requests music from the server
  handleClick () {
    console.log('Requesting', JSON.stringify(this.props.settings));
    ws.send(JSON.stringify(this.props.settings));
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