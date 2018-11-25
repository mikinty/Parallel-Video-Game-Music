import React from 'react';
import ReactDOM from 'react-dom';

import './styles/index.scss';

import * as CONST from './CONST';

// Components
import Title from './components/Title';
import RadioSelect from './components/RadioSelect';
import PlayButton from './components/PlayButton';

class App extends React.Component {
  componentDidMount () {
    document.title = CONST.TITLE;
  }

  render () {
    return (
      <div className='bg'>
        <div className='title'>
          <Title title={CONST.TITLE}/>
        </div>

        <div className='main'>
          <RadioSelect />
        </div>

        <div className='buttons'>
          <PlayButton />
        </div>
        
      </div>
    );
  }
}

ReactDOM.render(
  <App />,
  document.getElementById('app')
);

module.hot.accept();