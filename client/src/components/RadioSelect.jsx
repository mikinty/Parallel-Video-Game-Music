import React from 'react';
import * as CONST from '../CONST';

export default class RadioSelect extends React.Component {
  constructor (props) {
    super(props);

    this.state = {
      selected: 'Basic'
    };
  }

  // Called everytime we select an option
  handleChange (event) {
    this.setState({
      selected: event.target.value
    });

    let settings = CONST.SETTINGS_BASIC;

    switch (event.target.value) {
      case CONST.SETTINGS_BASIC.name:
        settings = CONST.SETTINGS_BASIC;
        break;
      case CONST.SETTINGS_JOURNEY.name:
        settings = CONST.SETTINGS_JOURNEY;
        break;
      case CONST.SETTINGS_SORROW.name:
        settings = CONST.SETTINGS_SORROW;
        break;
      case CONST.SETTINGS_BATTLE.name:
        settings = CONST.SETTINGS_BATTLE;
        break;
    }

    this.props.changeSettings(settings);

    console.log('Selected', event.target.value);
  }

  // TODO: put the radio options into separate files
  // Code adapted from https://codepen.io/anon/pen/NEMJvr
  render() {
    return (
      <div className="form">      
      <form>
        <fieldset className="form__options">
          <p className="form__answer"> 
            <input 
              type="radio" 
              name="match" 
              id="match_0" 
              value="basic" 
              checked={this.state.selected == CONST.SETTINGS_BASIC.name}
              onChange={this.handleChange.bind(this)} /> 
            <label htmlFor="match_0">
              <svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
                <title>Icon Guy</title>
                <g stroke="none" strokeWidth="1" fill="none" fillRule="evenodd">
                  <g id="Guy" stroke="#FFFFFF" strokeWidth="2">
                    <path d="M50,89 C28.4608948,89 11,71.5391052 11,50 C11,28.702089 28.8012779,11 50,11 C71.3811739,11 89,28.8647602 89,50 C89,71.5391052 71.5391052,89 50,89 Z" id="Oval"></path>
                    <path d="M34.5,59 C32.0147186,59 30,56.9852814 30,54.5 C30,53.1996039 30.5532818,51.9907899 31.5049437,51.1414086 C32.3241732,50.4102265 33.3788668,50 34.5,50 C36.9852814,50 39,52.0147186 39,54.5 C39,56.9852814 36.9852814,59 34.5,59 Z" id="eye"></path>
                    <path d="M65,59 C62.790861,59 61,57.209139 61,55 C61,53.844 61.4917357,52.7696523 62.3377558,52.0145589 C63.0660084,51.3645758 64.0033341,51 65,51 C67.209139,51 69,52.790861 69,55 C69,57.209139 67.209139,59 65,59 Z" id="eye"></path>
                    <path d="M13,39 C13,39 18.3404984,39.6508711 28,35 C37.6595016,30.3491289 40,26 40,26 C40,26 50.99493,36.4771587 58,38 C65.00507,39.5228413 75,42 86,36" id="Path-9"></path>
                    <path d="M40.0417765,73.6204199 C43.0857241,74.4024099 46.5428621,75 50,75 C53.4660267,75 57.0521869,74.3993329 60.2588177,73.6143844" id="Path-8"></path>
                  </g>
                </g>
              </svg>
              Basic
            </label> 
          </p>

          <p className="form__answer"> 
          <input 
              type="radio" 
              name="match" 
              id="match_2" 
              value="journey" 
              checked={this.state.selected == CONST.SETTINGS_JOURNEY.name}
              onChange={this.handleChange.bind(this)} /> 
            <label htmlFor="match_2">
              <svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
                <title>Icon Guy</title>
                <g stroke="none" strokeWidth="1" fill="none" fillRule="evenodd">
                  <g id="Guy" stroke="#FFFFFF" strokeWidth="2">
                    <path d="M50,89 C28.4608948,89 11,71.5391052 11,50 C11,28.702089 28.8012779,11 50,11 C71.3811739,11 89,28.8647602 89,50 C89,71.5391052 71.5391052,89 50,89 Z" id="Oval"></path>
                    <path d="M34.5,59 C32.0147186,59 30,56.9852814 30,54.5 C30,53.1996039 30.5532818,51.9907899 31.5049437,51.1414086 C32.3241732,50.4102265 33.3788668,50 34.5,50 C36.9852814,50 39,52.0147186 39,54.5 C39,56.9852814 36.9852814,59 34.5,59 Z" id="eye"></path>
                    <path d="M65,59 C62.790861,59 61,57.209139 61,55 C61,53.844 61.4917357,52.7696523 62.3377558,52.0145589 C63.0660084,51.3645758 64.0033341,51 65,51 C67.209139,51 69,52.790861 69,55 C69,57.209139 67.209139,59 65,59 Z" id="eye"></path>
                    <path d="M13,39 C13,39 18.3404984,39.6508711 28,35 C37.6595016,30.3491289 40,26 40,26 C40,26 50.99493,36.4771587 58,38 C65.00507,39.5228413 75,42 86,36" id="Path-9"></path>
                    <path d="M40.0417765,73.6204199 C43.0857241,74.4024099 46.5428621,75 50,75 C53.4660267,75 57.0521869,74.3993329 60.2588177,73.6143844" id="Path-8"></path>
                  </g>
                </g>
              </svg>
              Journey
            </label> 
          </p>

          <p className="form__answer"> 
          <input 
              type="radio" 
              name="match" 
              id="match_3" 
              value="sorrow" 
              checked={this.state.selected == CONST.SETTINGS_SORROW.name}
              onChange={this.handleChange.bind(this)} /> 
            <label htmlFor="match_3">
              <svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
                <title>Icon Guy</title>
                <g stroke="none" strokeWidth="1" fill="none" fillRule="evenodd">
                  <g id="Guy" stroke="#FFFFFF" strokeWidth="2">
                    <path d="M50,89 C28.4608948,89 11,71.5391052 11,50 C11,28.702089 28.8012779,11 50,11 C71.3811739,11 89,28.8647602 89,50 C89,71.5391052 71.5391052,89 50,89 Z" id="Oval"></path>
                    <path d="M34.5,59 C32.0147186,59 30,56.9852814 30,54.5 C30,53.1996039 30.5532818,51.9907899 31.5049437,51.1414086 C32.3241732,50.4102265 33.3788668,50 34.5,50 C36.9852814,50 39,52.0147186 39,54.5 C39,56.9852814 36.9852814,59 34.5,59 Z" id="eye"></path>
                    <path d="M65,59 C62.790861,59 61,57.209139 61,55 C61,53.844 61.4917357,52.7696523 62.3377558,52.0145589 C63.0660084,51.3645758 64.0033341,51 65,51 C67.209139,51 69,52.790861 69,55 C69,57.209139 67.209139,59 65,59 Z" id="eye"></path>
                    <path d="M13,39 C13,39 18.3404984,39.6508711 28,35 C37.6595016,30.3491289 40,26 40,26 C40,26 50.99493,36.4771587 58,38 C65.00507,39.5228413 75,42 86,36" id="Path-9"></path>
                    <path d="M40.0417765,73.6204199 C43.0857241,74.4024099 46.5428621,75 50,75 C53.4660267,75 57.0521869,74.3993329 60.2588177,73.6143844" id="Path-8"></path>
                  </g>
                </g>
              </svg>
              Sorrow
            </label> 
          </p>

          <p className="form__answer"> 
          <input 
              type="radio" 
              name="match" 
              id="match_4" 
              value="battle" 
              checked={this.state.selected == CONST.SETTINGS_BATTLE.name}
              onChange={this.handleChange.bind(this)} /> 
            <label htmlFor="match_4">
              <svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
                <title>Icon Guy</title>
                <g stroke="none" strokeWidth="1" fill="none" fillRule="evenodd">
                  <g id="Guy" stroke="#FFFFFF" strokeWidth="2">
                    <path d="M50,89 C28.4608948,89 11,71.5391052 11,50 C11,28.702089 28.8012779,11 50,11 C71.3811739,11 89,28.8647602 89,50 C89,71.5391052 71.5391052,89 50,89 Z" id="Oval"></path>
                    <path d="M34.5,59 C32.0147186,59 30,56.9852814 30,54.5 C30,53.1996039 30.5532818,51.9907899 31.5049437,51.1414086 C32.3241732,50.4102265 33.3788668,50 34.5,50 C36.9852814,50 39,52.0147186 39,54.5 C39,56.9852814 36.9852814,59 34.5,59 Z" id="eye"></path>
                    <path d="M65,59 C62.790861,59 61,57.209139 61,55 C61,53.844 61.4917357,52.7696523 62.3377558,52.0145589 C63.0660084,51.3645758 64.0033341,51 65,51 C67.209139,51 69,52.790861 69,55 C69,57.209139 67.209139,59 65,59 Z" id="eye"></path>
                    <path d="M13,39 C13,39 18.3404984,39.6508711 28,35 C37.6595016,30.3491289 40,26 40,26 C40,26 50.99493,36.4771587 58,38 C65.00507,39.5228413 75,42 86,36" id="Path-9"></path>
                    <path d="M40.0417765,73.6204199 C43.0857241,74.4024099 46.5428621,75 50,75 C53.4660267,75 57.0521869,74.3993329 60.2588177,73.6143844" id="Path-8"></path>
                  </g>
                </g>
              </svg>
              Battle
            </label> 
            </p>
        </fieldset>
      </form>
    </div>
    );
  }
}