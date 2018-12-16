import React from 'react';
import Tone from 'tone';
import * as SECRET from '../KEY';
import * as CONST from '../CONST';
import * as DATA from '../DATA';

// Do Tone js setup here

// 10 voice synth
const SYNTHS = [
  [
    new Tone.Synth({
      oscillator: {
        volume: CONST.VOLUME_CHORD
      }
    }),
    new Tone.Synth({
      oscillator: {
        volume: CONST.VOLUME_CHORD
      }
    }),
    new Tone.Synth({
      oscillator: {
        volume: CONST.VOLUME_CHORD
      }
    })
  ],
  [
    new Tone.Synth({
      oscillator: {
        type : 'sine',
        volume: CONST.VOLUME
      },
      envelope : {
        attack : CONST.ADSR.A,
        decay : CONST.ADSR.D,
        sustain : CONST.ADSR.S,
        release : CONST.ADSR.R
      }
    }),
    new Tone.Synth(),
    new Tone.Synth()
  ],
  [
    new Tone.Synth({
      oscillator: {
        type : 'sine',
        volume: CONST.VOLUME
      },
      envelope : {
        attack : CONST.ADSR.A,
        decay : CONST.ADSR.D,
        sustain : CONST.ADSR.S,
        release : CONST.ADSR.R
      }
    }),
    new Tone.Synth(),
    new Tone.Synth()
  ],
  [
    new Tone.Synth({
      oscillator: {
        type : 'sine',
        volume: CONST.VOLUME
      },
      envelope : {
        attack : CONST.ADSR.A,
        decay : CONST.ADSR.D,
        sustain : CONST.ADSR.S,
        release : CONST.ADSR.R
      }
    }),
    new Tone.Synth(),
    new Tone.Synth()
  ],
  [
    new Tone.Synth({
      oscillator: {
        type : 'sine',
        volume: CONST.VOLUME
      },
      envelope : {
        attack : CONST.ADSR.A,
        decay : CONST.ADSR.D,
        sustain : CONST.ADSR.S,
        release : CONST.ADSR.R
      }
    }),
    new Tone.Synth(),
    new Tone.Synth()
  ],
  [
    new Tone.Synth({
      oscillator: {
        type : 'sine',
        volume: CONST.VOLUME
      },
      envelope : {
        attack : CONST.ADSR.A,
        decay : CONST.ADSR.D,
        sustain : CONST.ADSR.S,
        release : CONST.ADSR.R
      }
    }),
    new Tone.Synth(),
    new Tone.Synth()
  ],
  [
    new Tone.Synth({
      oscillator: {
        type : 'sine',
        volume: CONST.VOLUME
      },
      envelope : {
        attack : CONST.ADSR.A,
        decay : CONST.ADSR.D,
        sustain : CONST.ADSR.S,
        release : CONST.ADSR.R
      }
    }),
    new Tone.Synth(),
    new Tone.Synth()
  ],
  [
    new Tone.Synth({
      oscillator: {
        type : 'sine',
        volume: CONST.VOLUME
      },
      envelope : {
        attack : CONST.ADSR.A,
        decay : CONST.ADSR.D,
        sustain : CONST.ADSR.S,
        release : CONST.ADSR.R
      }
    }),
    new Tone.Synth(),
    new Tone.Synth()
  ],
  [
    new Tone.Synth({
      oscillator: {
        type : 'sine',
        volume: CONST.VOLUME
      },
      envelope : {
        attack : CONST.ADSR.A,
        decay : CONST.ADSR.D,
        sustain : CONST.ADSR.S,
        release : CONST.ADSR.R
      }
    }),
    new Tone.Synth(),
    new Tone.Synth()
  ],
  [
    new Tone.Synth({
      oscillator: {
        type : 'sine',
        volume: CONST.VOLUME
      },
      envelope : {
        attack : CONST.ADSR.A,
        decay : CONST.ADSR.D,
        sustain : CONST.ADSR.S,
        release : CONST.ADSR.R
      }
    }),
    new Tone.Synth(),
    new Tone.Synth()
  ]
];

var lpf = new Tone.Filter({
  type: 'lowpass',
  frequency: CONST.LOWPASS_FREQ
}).toMaster();

var hpf = new Tone.Filter({
  type: 'highpass',
  frequency: CONST.HIGHPASS_FREQ
}).connect(lpf);

// connect to speakers
SYNTHS.forEach(s => {
  s[0].connect(hpf);
  s[1].connect(hpf);
  s[2].connect(hpf); 
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
  Tone.Transport.cancel();
  Tone.Transport.seconds = 0;

  // parse through the notes we are getting  
  for (let j = 0; j < notes.length; j++) {
    var currTime = 0;

    for (let i = 0; i < notes[j].length; i++) { 
      console.log(i, notes[j][i]);

      let note = notes[j][i][0];
      let duration = notes[j][i][1];

      // Schedule the music we just received
      if (note == CONST.REST_NOTE) {
        // rest
        Tone.Transport.scheduleOnce((time) => {
          console.log('rest');
          SYNTHS[j][0].triggerRelease(time);
        }, currTime);
      } else if (note < CONST.REST_NOTE) {
        // note
        Tone.Transport.scheduleOnce((time) => {
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
          Tone.Transport.scheduleOnce((time) => {
            console.log('chords playing', tempNotes, CONST.NOTE_DURATIONS[duration]);  

            SYNTHS[j][i].triggerAttackRelease (
              tempNotes[i],
              CONST.NOTE_DURATIONS[duration],
              time
            );
          }, currTime);
        }
      }

      currTime = currTime + Tone.Time(CONST.NOTE_DURATIONS[notes[j][i][1]]).toSeconds();
    }
  }

  console.log('Start playing at', Tone.Transport.bpm.value);
  console.log(Tone.Transport);
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

    this.state = {
      genre: CONST.SETTINGS_BASIC.name
    };
  }

  // Called when we click play
  // Requests music from the server
  handleClick () {
    console.log('Requesting', JSON.stringify(this.props.settings));

    Tone.Transport.bpm.value = this.props.settings.tempo;

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