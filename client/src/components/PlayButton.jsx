import React from 'react';

export default class RadioSelect extends React.Component {
  constructor () {
    super();
  }

  // Called when we click play
  handleClick () {
    console.log('Clicked button!');
  }

  render() {
    return (
      <div class='playBut' onClick={this.handleClick}>
        <svg version="1.1"
          xmlns="http://www.w3.org/2000/svg"
          x="0px" y="0px" width="120px" height="120px" viewBox="0 0 213.7 213.7" enableBackground="new 0 0 213.7 213.7"
          space="preserve">
        
          <polygon class='triangle' id="XMLID_18_" fill="none" stroke-width="7" strokeLinecap="round" strokeLinejoin="round" stroke-miterlimit="10" points="73.5,62.5 148.5,105.8 73.5,149.1 "/>
            
          <circle class='circle' id="XMLID_17_" fill="none"  stroke-width="7" strokeLinecap="round" strokeLinejoin="round" stroke-miterlimit="10" cx="106.8" cy="106.8" r="103.3"/>
        </svg>
      </div>
    );
  }
}