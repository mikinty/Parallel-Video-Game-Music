import React from 'react';
import Tone from 'tone';
import * as SECRET from '../KEY';

const synth = new Tone.Synth();
const ws = new WebSocket('ws://' + SECRET.SECRET_ADDRESS);

// listen to websocket events
ws.onmessage = (event) => {
  console.log(event);
}

export default class RadioSelect extends React.Component {
  constructor () {
    super();

    // setup our synthesizer
    synth.toMaster();
  }
  

  // Called when we click play
  handleClick () {
    synth.triggerAttackRelease("C4", 0.25);
    ws.send('TODD MOWRY IS THE BEST');
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