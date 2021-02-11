import React from "react";
import Typist from 'react-typist';
import LazyLoad from 'react-lazyload';

// reactstrap components
import {
  Container,
  Row,
  Col
} from "reactstrap";

export default function Basics() {
  return (
    <div className="section section-basic" id="basic-elements">
      <img
        alt="..."
        className="path"
        src={require("assets/img/path1.png").default}
      />
      <Container>
        <Row>
          <Col md="12">
            <h1 className="title">Most Text-Based ML Systems Are Broken.</h1>
            <p className="main-text">Unlike human writing, modern computers can encode any given piece of text with a near-infinite number of unique logical encodings. This is derived from the fact that common language encodings, such as <a href="https://unicode.org" target="_blank" rel="noreferrer">Unicode</a>, provide methods to create differences between the logical encoding of a string and its visual rending.</p>
            <p className="main-text pt-3 pb-5">Since text-based ML models, and most NLP systems more broadly, operate upon the logical encoding of text as inputs, the difference between logical encoding and visual rendering can be used to deceive users and adversarially control the output of these systems.</p>
            <LazyLoad><h1 className="text-center"><code><Typist stdTypingDelay={50} avgTypingDelay={100}>Send money to 1234 --> Send money to 2314</Typist></code></h1></LazyLoad>

            <h1 className="title pt-5 mt-5">These Differences Can Manipulate NLP Systems.</h1>
            <p className="main-text"></p>
          </Col>
        </Row>
      </Container>
    </div>
  );
}