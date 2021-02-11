import React from "react";

// core components
import IndexNavbar from "components/Navbars/IndexNavbar.js";
import Footer from "components/Footer/Footer.js";

// reactstrap components
import {
  Container,
  Row,
  Col,
} from "reactstrap";

export default function Validate() {
  return (
    <>
      <IndexNavbar />
      <div className="wrapper">
        <div className="page-header">
          <img
            alt="..."
            className="dots"
            src={require("assets/img/dots.png").default}
          />
          <img
            alt="..."
            className="path"
            src={require("assets/img/path2.png").default}
          />
          <Container className="below-nav">
              <Row>
                  <Col md={12} className="text-center">
                    <h1><b>Validate</b></h1>
                  </Col>
              </Row>
        </Container>
      </div>
      <Footer />
    </div>
    </>
  );
}
