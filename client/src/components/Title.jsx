import React from 'react';

export default class Title extends React.Component {
  constructor (props) {
    super();

    this.state = {
      title: props.title
    };
  }

  render() {
    return (
      <h1>{this.state.title}</h1>
    );
  }
}