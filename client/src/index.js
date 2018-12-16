import React from 'react';
import ReactDOM from 'react-dom';

import './styles/index.scss';

import * as CONST from './CONST';

// Components
import Title from './components/Title';
import RadioSelect from './components/RadioSelect';
import PlayButton from './components/PlayButton';

class App extends React.Component {
  constructor (props) {
    super(props);

    this.state = {
      settings: CONST.SETTINGS_BASIC
    };
  }
  componentDidMount () {
    document.title = CONST.TITLE;
  }

  changeSettings (newSettings) {
    console.log('new setting', newSettings);
    this.setState({
      settings: newSettings
    });
  }

  getSettings () {
    return this.state.settings;
  }

  render () {
    return (
      <div className='bg'>
        <div className='title'>
          <Title title={CONST.TITLE}/>
        </div>

        <div className='main'>
          <RadioSelect changeSettings={this.changeSettings.bind(this)} />
        </div>

        <div className='buttons'>
          <PlayButton settings={this.state.settings} />
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