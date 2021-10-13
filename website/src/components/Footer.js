import React from "react";
// reactstrap components
import {
  Button,
  Container,
  Row,
  Col,
  UncontrolledTooltip,
} from "reactstrap";

const Footer = React.forwardRef((props, ref) => {
  return (
    <footer {...props} ref = {ref} className="footer">
      <style>{`
        #boucher {
          background-image: url('${require("assets/img/boucher.jpg").default}') !important;
        }
        #shumailov {
          background-image: url('${require("assets/img/shumailov.jpg").default}') !important;
        }
        #anderson {
          background-image: url('${require("assets/img/anderson.jpg").default}') !important;
        }
        #papernot {
          background-image: url('${require("assets/img/papernot.jpg").default}') !important;
        }
      `}</style>
      <Container>
        <Row>
          <Col md="9">
            Produced by researchers from The University of Cambridge and The University of Toronto.<br />
            Website by <a href="https://www.cl.cam.ac.uk/~ndb40">Nicholas Boucher</a> with thanks to <a href="https://github.com/creativetimofficial/blk-design-system-react" rel="noreferrer">Blk• React</a> and <a href="https://www.srcf.net" rel="noreferrer">SRCF</a>.<br />
            <br />
            &copy;&nbsp;{new Date().getFullYear()}
          </Col>
          <Col md="3">
            <h4>Authors:</h4>
            <div className="btn-wrapper profile">
              <Button
                className="btn-icon btn-neutral btn-round btn-simple bg-img-contain"
                color="default"
                href="https://www.cl.cam.ac.uk/~ndb40"
                id="boucher"
                target="_blank"
              />
              <UncontrolledTooltip delay={0} target="boucher">
                Nicholas Boucher
              </UncontrolledTooltip>
              <Button
                className="btn-icon btn-neutral btn-round btn-simple bg-img-contain"
                color="default"
                href="https://www.cl.cam.ac.uk/~is410"
                id="shumailov"
                target="_blank"
              />
              <UncontrolledTooltip delay={0} target="shumailov">
                Ilia Shumailov
              </UncontrolledTooltip>
              <Button
                className="btn-icon btn-neutral btn-round btn-simple bg-img-contain"
                color="default"
                href="https://www.cl.cam.ac.uk/~rja14"
                id="anderson"
                target="_blank"
              />
              <UncontrolledTooltip delay={0} target="anderson">
                Ross Anderson
              </UncontrolledTooltip>
              <Button
                className="btn-icon btn-neutral btn-round btn-simple bg-img-contain"
                color="default"
                href="https://www.papernot.fr"
                id="papernot"
                target="_blank"
              />
              <UncontrolledTooltip delay={0} target="papernot">
              Nicolas Papernot
              </UncontrolledTooltip>
            </div>
          </Col>
        </Row>
      </Container>
    </footer>
  );
});


export default Footer;