import React from "react";
// reactstrap components
import {
  Button,
  Container,
  Row,
  Col,
  UncontrolledTooltip,
} from "reactstrap";

export default function Footer() {
  return (
    <footer className="footer">
      <Container>
        <Row>
          <Col md="9">
            Produced by researchers from The University of Cambridge and The University of Toronto.<br />
            Website by <a href="https://www.cl.cam.ac.uk/~ndb40">Nicholas Boucher</a> with thanks to <a href="https://github.com/creativetimofficial/blk-design-system-react">Blk• React</a>.<br />
            <br />
            &copy;&nbsp;2021
          </Col>
          <Col md="3">
            <h4>Authors:</h4>
            <div className="btn-wrapper profile">
              <Button
                className="btn-icon btn-neutral btn-round btn-simple"
                color="default"
                href="https://www.cl.cam.ac.uk/~ndb40"
                id="boucher"
                target="_blank"
              >
                <i className="fa fa-user" />
              </Button>
              <UncontrolledTooltip delay={0} target="boucher">
                Nicholas Boucher
              </UncontrolledTooltip>
              <Button
                className="btn-icon btn-neutral btn-round btn-simple"
                color="default"
                href="https://www.cl.cam.ac.uk/~is410"
                id="shumailov"
                target="_blank"
              >
                <i className="fa fa-user" />
              </Button>
              <UncontrolledTooltip delay={0} target="shumailov">
                Ilia Shumailov
              </UncontrolledTooltip>
              <Button
                className="btn-icon btn-neutral btn-round btn-simple"
                color="default"
                href="https://www.cl.cam.ac.uk/~rja14"
                id="anderson"
                target="_blank"
              >
                <i className="fa fa-user" />
              </Button>
              <UncontrolledTooltip delay={0} target="anderson">
                Ross Anderson
              </UncontrolledTooltip>
              <Button
                className="btn-icon btn-neutral btn-round btn-simple"
                color="default"
                href="https://www.papernot.fr"
                id="papernot"
                target="_blank"
              >
                <i className="fa fa-user" />
              </Button>
              <UncontrolledTooltip delay={0} target="papernot">
              Nicolas Papernot
              </UncontrolledTooltip>
            </div>
          </Col>
        </Row>
      </Container>
    </footer>
  );
}
