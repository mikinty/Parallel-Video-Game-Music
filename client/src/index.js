import React from 'react';
import ReactDOM from 'react-dom';

import './styles/index.scss';

import * as CONST from './CONST';

// Components
import RadioSelect from './components/RadioSelect';

ReactDOM.render(
  <div className='bg'>
    <div className='title'>
      <h1>{CONST.TITLE}</h1>
    </div>

    <div className='main'>
      <RadioSelect />
    </div>

    <div className='buttons'>
    <a href='#'class='playBut'>
      
      <svg version="1.1"
        xmlns="http://www.w3.org/2000/svg"
        x="0px" y="0px" width="120px" height="120px" viewBox="0 0 213.7 213.7" enable-background="new 0 0 213.7 213.7"
        space="preserve">
      
        <polygon class='triangle' id="XMLID_18_" fill="none" stroke-width="7" stroke-linecap="round" stroke-linejoin="round" stroke-miterlimit="10" points="73.5,62.5 148.5,105.8 73.5,149.1 "/>
          
        <circle class='circle' id="XMLID_17_" fill="none"  stroke-width="7" stroke-linecap="round" stroke-linejoin="round" stroke-miterlimit="10" cx="106.8" cy="106.8" r="103.3"/>
      </svg>
    </a>
    </div>
    
  </div>,
  document.getElementById('app')
);

module.hot.accept();