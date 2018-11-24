import React from 'react';

export default class RadioSelect extends React.Component {
  constructor () {
    super();

    this.state = {
      selected: 'Basic'
    };
  }

  // Called everytime we select an option
  handleChange (event) {
    this.setState({
      selected: event.target.value
    });
  }

  // TODO: put the radio options into separate files
  render() {
    return (
      <form>
        <div className="radioOption">
          <label>
            <input 
              type="radio" 
              value="Basic" 
              checked={this.state.selected == 'Basic'}
              onChange={this.handleChange.bind(this)}/>
            Basic
          </label>
        </div>

        <div className="radioOption">
          <label>
            <input 
              type="radio" 
              value="Happy" 
              checked={this.state.selected == 'Happy'}
              onChange={this.handleChange.bind(this)}/>
            Happy
          </label>
        </div>
      </form>
    );
  }
}