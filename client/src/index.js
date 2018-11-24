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
      <a id="play-video" className="video-play-button" href="#">
        <span></span>
      </a>
    </div>
    
  </div>,
  document.getElementById('app')
);

module.hot.accept();